import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import "./styles.css";

type Point = { x: number; y: number; t: number };

type Shape = {
  id: number;
  points: Point[];
  center: Point;
  width: number;
  height: number;
};

type Particle = {
  x: number; y: number; vx: number; vy: number;
  life: number; maxLife: number; size: number;
};

type HandSlot = {
  key: "Left" | "Right";
  landmarks: NormalizedLandmark[];
  gesture: "draw" | "erase" | "idle";
  index: Point;
  middle: Point;
  lastPoint: Point | null;
  drawing: boolean;
  shapeId: number | null;
  dragOffset: Point;
};

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MAX_POINTS = 1800;
const SMOOTHING = 0.72;

function distance(a: NormalizedLandmark, b: NormalizedLandmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function angle(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const m1 = Math.hypot(ab.x, ab.y, ab.z);
  const m2 = Math.hypot(cb.x, cb.y, cb.z);
  if (!m1 || !m2) return 0;
  return Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180 / Math.PI;
}

function isIndexOnlyPose(hand: NormalizedLandmark[]) {
  if (hand.length < 21) return false;
  const wrist = hand[0];
  const indexExtended =
    distance(hand[8], wrist) > distance(hand[6], wrist) * 1.10 &&
    angle(hand[5], hand[6], hand[7]) > 150 &&
    angle(hand[6], hand[7], hand[8]) > 140;

  const middleFolded = distance(hand[12], wrist) < distance(hand[10], wrist) * 1.08;
  const ringFolded = distance(hand[16], wrist) < distance(hand[14], wrist) * 1.08;
  const pinkyFolded = distance(hand[20], wrist) < distance(hand[18], wrist) * 1.08;
  const thumbNearPalm = distance(hand[4], hand[5]) < distance(hand[8], wrist) * 0.72;

  return indexExtended && middleFolded && ringFolded && pinkyFolded && thumbNearPalm;
}

function isTwoFingerErasePose(hand: NormalizedLandmark[]) {
  if (hand.length < 21) return false;
  const wrist = hand[0];

  const indexExtended =
    distance(hand[8], wrist) > distance(hand[6], wrist) * 1.08 &&
    angle(hand[5], hand[6], hand[7]) > 145 &&
    angle(hand[6], hand[7], hand[8]) > 135;

  const middleExtended =
    distance(hand[12], wrist) > distance(hand[10], wrist) * 1.08 &&
    angle(hand[9], hand[10], hand[11]) > 145 &&
    angle(hand[10], hand[11], hand[12]) > 135;

  const ringFolded = distance(hand[16], wrist) < distance(hand[14], wrist) * 1.10;
  const pinkyFolded = distance(hand[20], wrist) < distance(hand[18], wrist) * 1.10;

  return indexExtended && middleExtended && ringFolded && pinkyFolded;
}

function pathLength(points: Point[]) {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  return total;
}

function polygonArea(points: Point[]) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return Math.abs(area) / 2;
}

function bounds(points: Point[]) {
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
}

function centerOf(points: Point[]): Point {
  const b = bounds(points);
  return { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2, t: performance.now() };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectorRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);

  const strokesRef = useRef<Point[][]>([]);
  const shapesRef = useRef<Shape[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const handsRef = useRef<HandSlot[]>([]);
  const nextShapeIdRef = useRef(1);
  const lastTimeRef = useRef(-1);

  const [ready, setReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [leftGesture, setLeftGesture] = useState("IDLE");
  const [rightGesture, setRightGesture] = useState("IDLE");
  const [error, setError] = useState("");

  const resizeCanvas = useCallback(() => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = video.videoWidth || window.innerWidth;
    const height = video.videoHeight || window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const detector = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
        if (!cancelled) {
          detectorRef.current = detector;
          setReady(true);
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setError("MediaPipe yuklanmadi. Internet aloqasini tekshiring.");
      }
    })();
    return () => {
      cancelled = true;
      detectorRef.current?.close();
    };
  }, []);

  useEffect(() => {
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, [resizeCanvas]);

  const clearDrawing = useCallback(() => {
    strokesRef.current = [];
    shapesRef.current = [];
    particlesRef.current = [];
    handsRef.current.forEach(h => {
      h.lastPoint = null; h.drawing = false; h.shapeId = null;
    });
  }, []);

  const addParticles = useCallback((x: number, y: number) => {
    for (let i = 0; i < 2; i++) {
      particlesRef.current.push({
        x, y, vx: (Math.random() - .5) * 2.8, vy: (Math.random() - .5) * 2.8,
        life: 1, maxLife: .45 + Math.random() * .55, size: 1 + Math.random() * 2.5
      });
    }
    if (particlesRef.current.length > 400) particlesRef.current.splice(0, particlesRef.current.length - 400);
  }, []);

  const eraseAt = useCallback((x: number, y: number) => {
    const r = 42, r2 = r * r;
    const next: Point[][] = [];
    for (const stroke of strokesRef.current) {
      let segment: Point[] = [];
      for (const p of stroke) {
        if ((p.x - x) ** 2 + (p.y - y) ** 2 <= r2) {
          if (segment.length > 1) next.push(segment);
          segment = [];
        } else segment.push(p);
      }
      if (segment.length > 1) next.push(segment);
    }
    strokesRef.current = next;

    shapesRef.current = shapesRef.current.filter(shape => {
      const b = bounds(shape.points);
      return !(x >= b.minX - r && x <= b.maxX + r && y >= b.minY - r && y <= b.maxY + r);
    });
  }, []);

  const tryCloseStroke = useCallback((stroke: Point[]) => {
    if (stroke.length < 22 || pathLength(stroke) < 130) return false;
    const first = stroke[0], last = stroke[stroke.length - 1];
    const closeDistance = Math.hypot(last.x - first.x, last.y - first.y);
    const b = bounds(stroke);
    if (closeDistance > Math.max(34, Math.min(b.width, b.height) * 0.22)) return false;
    if (b.width < 45 || b.height < 45) return false;
    if (polygonArea(stroke) < 900) return false;

    const shape: Shape = {
      id: nextShapeIdRef.current++,
      points: stroke.slice(),
      center: centerOf(stroke),
      width: b.width,
      height: b.height,
    };
    shapesRef.current.push(shape);
    strokesRef.current.pop();
    return true;
  }, []);

  const findShapeAt = useCallback((x: number, y: number) => {
    for (let i = shapesRef.current.length - 1; i >= 0; i--) {
      const s = shapesRef.current[i], b = bounds(s.points);
      const pad = 28;
      if (x >= b.minX - pad && x <= b.maxX + pad && y >= b.minY - pad && y <= b.maxY + pad) return s;
    }
    return null;
  }, []);

  const moveShape = useCallback((shape: Shape, dx: number, dy: number) => {
    shape.points = shape.points.map(p => ({ ...p, x: p.x + dx, y: p.y + dy }));
    shape.center = { ...shape.center, x: shape.center.x + dx, y: shape.center.y + dy };
  }, []);

  const drawShape = (ctx: CanvasRenderingContext2D, shape: Shape) => {
    if (shape.points.length < 2) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(shape.points[0].x, shape.points[0].y);
    for (let i = 1; i < shape.points.length; i++) ctx.lineTo(shape.points[i].x, shape.points[i].y);
    ctx.closePath();
    ctx.strokeStyle = "rgba(50,255,125,.35)";
    ctx.lineWidth = 16;
    ctx.shadowBlur = 32;
    ctx.shadowColor = "#35ff86";
    ctx.stroke();
    ctx.strokeStyle = "#35ff86";
    ctx.lineWidth = 4;
    ctx.shadowBlur = 18;
    ctx.stroke();
    ctx.strokeStyle = "rgba(235,255,245,.95)";
    ctx.lineWidth = 1.2;
    ctx.shadowBlur = 0;
    ctx.stroke();
    ctx.restore();
  };

  const render = useCallback(() => {
    const canvas = canvasRef.current, video = videoRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);

    const toScreen = (lm: NormalizedLandmark) => {
      if (!video || !video.videoWidth) return { x: lm.x * width, y: lm.y * height };
      const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
      const rw = video.videoWidth * scale, rh = video.videoHeight * scale;
      return { x: (1 - lm.x) * rw - (rw - width) / 2, y: lm.y * rh - (rh - height) / 2 };
    };

    const connections: [number, number][] = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
    ];

    for (const hand of handsRef.current) {
      const pts = hand.landmarks.map(toScreen);
      ctx.save();
      ctx.beginPath();
      for (const [a,b] of connections) { ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); }
      ctx.strokeStyle = "rgba(0,234,255,.28)";
      ctx.lineWidth = 10; ctx.lineCap = "round"; ctx.shadowBlur = 25; ctx.shadowColor = "#00eaff"; ctx.stroke();
      ctx.strokeStyle = "#00eaff"; ctx.lineWidth = 3.2; ctx.shadowBlur = 12; ctx.stroke();
      for (const p of pts) {
        ctx.beginPath(); ctx.arc(p.x,p.y,3.8,0,Math.PI*2); ctx.fillStyle="#eaffff"; ctx.shadowBlur=14; ctx.fill();
      }
      ctx.restore();

      // The middle-finger erase cursor/indicator is intentionally hidden.
      // Erasing still works internally via hand.middle, but no red marker is drawn.
      if (hand.gesture !== "erase") {
        const cursor = hand.index;
        ctx.save();
        ctx.beginPath();
        ctx.arc(cursor.x, cursor.y, 8, 0, Math.PI*2);
        ctx.fillStyle = "#eaffff";
        ctx.strokeStyle = "rgba(0,234,255,.8)";
        ctx.lineWidth = 2; ctx.shadowBlur = 20; ctx.shadowColor = "#00eaff";
        ctx.fill(); ctx.stroke(); ctx.restore();
      }
    }

    for (const stroke of strokesRef.current) {
      if (stroke.length < 2) continue;
      ctx.save();
      ctx.beginPath(); ctx.moveTo(stroke[0].x, stroke[0].y);
      for (let i=1;i<stroke.length;i++) ctx.lineTo(stroke[i].x, stroke[i].y);
      ctx.strokeStyle="rgba(0,220,255,.35)"; ctx.lineWidth=14; ctx.lineCap="round"; ctx.lineJoin="round"; ctx.shadowBlur=30; ctx.shadowColor="#00e5ff"; ctx.stroke();
      ctx.strokeStyle="#00eaff"; ctx.lineWidth=4; ctx.shadowBlur=18; ctx.stroke();
      ctx.strokeStyle="rgba(235,255,255,.9)"; ctx.lineWidth=1.15; ctx.shadowBlur=0; ctx.stroke();
      ctx.restore();
    }

    for (const shape of shapesRef.current) drawShape(ctx, shape);

    for (let i=particlesRef.current.length-1;i>=0;i--) {
      const p=particlesRef.current[i]; p.x+=p.vx; p.y+=p.vy; p.life-=.025;
      if(p.life<=0){particlesRef.current.splice(i,1);continue;}
      ctx.save(); ctx.globalAlpha=Math.min(1,p.life/p.maxLife); ctx.beginPath();
      ctx.arc(p.x,p.y,p.size,0,Math.PI*2); ctx.fillStyle="#cfffff"; ctx.shadowBlur=12; ctx.shadowColor="#00eaff"; ctx.fill(); ctx.restore();
    }
  }, []);

  const processFrame = useCallback(() => {
    const video=videoRef.current, detector=detectorRef.current;
    if (!video || !detector || video.readyState < 2) {
      rafRef.current=requestAnimationFrame(processFrame); return;
    }

    const now=performance.now();
    if(lastTimeRef.current<0 || now>lastTimeRef.current) {
      const result: HandLandmarkerResult = detector.detectForVideo(video,now);
      lastTimeRef.current=now;

      const canvas=canvasRef.current;
      if(canvas && result.landmarks.length) {
        const viewW=canvas.clientWidth, viewH=canvas.clientHeight;
        const scale=Math.max(viewW/video.videoWidth,viewH/video.videoHeight);
        const rw=video.videoWidth*scale,rh=video.videoHeight*scale;
        const cropX=(rw-viewW)/2,cropY=(rh-viewH)/2;

        const detected: HandSlot[] = result.landmarks.map((landmarks,index) => {
          const label=(result.handednesses[index]?.[0]?.categoryName || "").toLowerCase();
          const key: "Left"|"Right" = label.includes("left") ? "Left" : "Right";
          const old=handsRef.current.find(h=>h.key===key);
          const point=(lm:NormalizedLandmark):Point=>({
            x:(1-lm.x)*rw-cropX,y:lm.y*rh-cropY,t:now
          });
          const rawIndex=point(landmarks[8]), rawMiddle=point(landmarks[12]);
          const previous=old?.lastPoint;
          const indexPoint:Point = previous ? {
            x:previous.x*SMOOTHING+rawIndex.x*(1-SMOOTHING),
            y:previous.y*SMOOTHING+rawIndex.y*(1-SMOOTHING),t:now
          } : rawIndex;

          const middlePoint:Point=previous ? {
            x:previous.x*SMOOTHING+rawMiddle.x*(1-SMOOTHING),
            y:previous.y*SMOOTHING+rawMiddle.y*(1-SMOOTHING),t:now
          } : rawMiddle;

          const erase=isTwoFingerErasePose(landmarks);
          const draw=isIndexOnlyPose(landmarks);
          return {
            key, landmarks, gesture: erase ? "erase" : draw ? "draw" : "idle",
            index:indexPoint, middle:middlePoint,
            lastPoint:previous || null,
            drawing:old?.drawing || false,
            shapeId:old?.shapeId ?? null,
            dragOffset:old?.dragOffset || {x:0,y:0,t:now}
          };
        });

        // STRICT ROLES:
        // RIGHT HAND = drawing + erasing only.
        // LEFT HAND = moving completed green shapes only.
        for (const hand of detected) {
          if (hand.key === "Right") {
            // RIGHT HAND: ✌ = erase.
            if (hand.gesture === "erase") {
              eraseAt(hand.middle.x, hand.middle.y);
              hand.drawing = false;
              hand.lastPoint = null;
              hand.shapeId = null;
              continue;
            }

            // RIGHT HAND: ☝ = draw.
            // It can NOT move completed shapes.
            if (hand.gesture === "draw") {
              hand.shapeId = null;

              if (!hand.drawing) {
                strokesRef.current.push([hand.index]);
                hand.drawing = true;
              } else {
                const stroke = strokesRef.current[strokesRef.current.length - 1];
                const last = stroke?.[stroke.length - 1];

                if (
                  stroke &&
                  (!last ||
                    Math.hypot(
                      hand.index.x - last.x,
                      hand.index.y - last.y
                    ) > 1.2)
                ) {
                  stroke.push(hand.index);
                  tryCloseStroke(stroke);
                }
              }

              hand.lastPoint = hand.index;
              addParticles(hand.index.x, hand.index.y);
            } else {
              hand.drawing = false;
              hand.lastPoint = null;
              hand.shapeId = null;
            }
          } else {
            // LEFT HAND: ONLY move completed green shapes.
            // It can never create a stroke and can never erase.
            hand.drawing = false;

            if (hand.gesture === "draw") {
              if (hand.shapeId !== null) {
                const shape = shapesRef.current.find(
                  s => s.id === hand.shapeId
                );

                if (shape) {
                  const dx =
                    hand.index.x -
                    (hand.lastPoint?.x ?? hand.index.x);
                  const dy =
                    hand.index.y -
                    (hand.lastPoint?.y ?? hand.index.y);

                  moveShape(shape, dx, dy);
                } else {
                  hand.shapeId = null;
                }
              } else {
                const hit = findShapeAt(
                  hand.index.x,
                  hand.index.y
                );

                if (hit) {
                  const otherOwns = detected.some(
                    h => h !== hand && h.shapeId === hit.id
                  );

                  if (!otherOwns) {
                    hand.shapeId = hit.id;
                  }
                }
              }

              hand.lastPoint = hand.index;
            } else {
              // Any other left-hand pose releases the shape.
              hand.shapeId = null;
              hand.lastPoint = null;
            }
          }
        }

        let total=strokesRef.current.reduce((n,s)=>n+s.length,0);
        while(total>MAX_POINTS && strokesRef.current.length>1) {
          total-=strokesRef.current.shift()?.length || 0;
        }

        handsRef.current=detected;
        setLeftGesture(detected.find(h=>h.key==="Left")?.gesture.toUpperCase() || "NOT FOUND");
        setRightGesture(detected.find(h=>h.key==="Right")?.gesture.toUpperCase() || "NOT FOUND");
      } else {
        handsRef.current=[];
        setLeftGesture("NOT FOUND"); setRightGesture("NOT FOUND");
      }
    }

    render();
    rafRef.current=requestAnimationFrame(processFrame);
  }, [addParticles, eraseAt, findShapeAt, moveShape, render, tryCloseStroke]);

  async function startCamera() {
    try {
      setError("");
      const stream=await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},audio:false
      });
      streamRef.current=stream;
      const video=videoRef.current;
      if(!video) return;
      video.srcObject=stream;
      await new Promise<void>(resolve=>{video.onloadedmetadata=()=>resolve();});
      await video.play();
      resizeCanvas();
      setCameraOn(true);
      lastTimeRef.current=-1;
      rafRef.current=requestAnimationFrame(processFrame);
    } catch(e) {
      console.error(e);
      setError("Kameraga ruxsat berilmadi yoki kamera topilmadi. Browser permission-ni tekshiring.");
    }
  }

  function stopCamera() {
    if(rafRef.current!==null){cancelAnimationFrame(rafRef.current);rafRef.current=null;}
    streamRef.current?.getTracks().forEach(t=>t.stop());
    streamRef.current=null;
    if(videoRef.current) videoRef.current.srcObject=null;
    setCameraOn(false);
    handsRef.current=[];
    lastTimeRef.current=-1;
    setLeftGesture("IDLE"); setRightGesture("IDLE");
  }

  return (
    <main className="app">
      <video ref={videoRef} className="video" playsInline muted />
      <canvas ref={canvasRef} className="canvas" />

      <div className="topbar">
        <div>
          <div className="brand">NEON HAND</div>
          <div className="subtitle">RIGHT: DRAW + ERASE • LEFT: MOVE SHAPES</div>
        </div>
        <div className={`status ${cameraOn ? "drawing-status" : ""}`}>
          <span className={`dot ${ready ? "ready" : ""}`} />
          {!ready ? "LOADING MODEL..." : !cameraOn ? "CAMERA OFF" : "● TWO HAND TRACKING"}
        </div>
      </div>

      {cameraOn && (
        <div className="hand-status">
          <div>LEFT HAND: <b>{leftGesture}</b></div>
          <div>RIGHT HAND: <b>{rightGesture}</b></div>
        </div>
      )}

      {!cameraOn && (
        <button className="start-button" disabled={!ready} onClick={startCamera}>
          {ready ? "START CAMERA" : "LOADING..."}
        </button>
      )}

      {cameraOn && (
        <div className="controls">
          <button className="secondary" onClick={clearDrawing}>CLEAR</button>
          <button className="secondary" onClick={stopCamera}>STOP CAMERA</button>
        </div>
      )}

      <div className="hint">
        {cameraOn
          ? "O‘NG QO‘L: ☝ CHIZISH • ✌ O‘CHIRISH &nbsp; | &nbsp; CHAP QO‘L: ☝ TAYYOR YASHIL SHAKLNI SILJITISH"
          : "O‘NG QO‘L bilan chizing/o‘chiring • CHAP QO‘L bilan faqat tayyor yashil shaklni siljiting"}
      </div>

      {error && <div className="error">{error}</div>}
    </main>
  );
}
