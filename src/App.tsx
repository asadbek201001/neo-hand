import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import "./styles.css";

type Point = {
  x: number;
  y: number;
  t: number;
};

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
};

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MAX_POINTS = 1800;

// Higher = smoother, but slightly more latency.
const SMOOTHING = 0.72;

// Require the index finger to be clearly extended.
const INDEX_EXTENSION_RATIO = 1.10;

// Other fingers must be clearly shorter/folded.
const FOLDED_RATIO = 1.08;

function distance(a: NormalizedLandmark, b: NormalizedLandmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function angle(
  a: NormalizedLandmark,
  b: NormalizedLandmark,
  c: NormalizedLandmark
) {
  const ab = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z,
  };

  const cb = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: c.z - b.z,
  };

  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const magAB = Math.hypot(ab.x, ab.y, ab.z);
  const magCB = Math.hypot(cb.x, cb.y, cb.z);

  if (!magAB || !magCB) return 0;

  const cosine = Math.min(1, Math.max(-1, dot / (magAB * magCB)));

  return Math.acos(cosine) * (180 / Math.PI);
}

/**
 * Returns true only when the pose is approximately:
 *
 *       ☝
 *
 * index finger = extended
 * middle/ring/pinky = folded
 * thumb = folded/near palm
 */
function isIndexOnlyPose(hand: NormalizedLandmark[]) {
  if (hand.length < 21) return false;

  const wrist = hand[0];

  const thumbTip = hand[4];
  const indexMcp = hand[5];
  const indexPip = hand[6];
  const indexDip = hand[7];
  const indexTip = hand[8];

  const middlePip = hand[10];
  const middleTip = hand[12];

  const ringPip = hand[14];
  const ringTip = hand[16];

  const pinkyPip = hand[18];
  const pinkyTip = hand[20];

  // Index must be physically farther from the wrist than its PIP.
  const indexTipWrist = distance(indexTip, wrist);
  const indexPipWrist = distance(indexPip, wrist);

  const indexExtended =
    indexTipWrist > indexPipWrist * INDEX_EXTENSION_RATIO &&
    angle(indexMcp, indexPip, indexDip) > 155 &&
    angle(indexPip, indexDip, indexTip) > 145;

  // Other three long fingers must be folded.
  const middleFolded =
    distance(middleTip, wrist) < distance(middlePip, wrist) * FOLDED_RATIO;

  const ringFolded =
    distance(ringTip, wrist) < distance(ringPip, wrist) * FOLDED_RATIO;

  const pinkyFolded =
    distance(pinkyTip, wrist) < distance(pinkyPip, wrist) * FOLDED_RATIO;

  // Thumb should stay near the palm/index-MCP area.
  const thumbNearPalm =
    distance(thumbTip, indexMcp) <
    distance(indexTip, wrist) * 0.72;

  return (
    indexExtended &&
    middleFolded &&
    ringFolded &&
    pinkyFolded &&
    thumbNearPalm
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const detectorRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);

  const strokesRef = useRef<Point[][]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const lastPointRef = useRef<Point | null>(null);
  const eraserPointRef = useRef<Point | null>(null);
  const handLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const lastTimeRef = useRef(-1);
  // True only while the current hand gesture is continuously drawing.
  // This is deliberately separate from lastPointRef: lastPointRef is
  // also used for visual smoothing and must not decide stroke continuity.
  const drawingSessionRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [indexOnly, setIndexOnly] = useState(false);
  const [erasing, setErasing] = useState(false);
  const [error, setError] = useState("");

  const resizeCanvas = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = video.videoWidth || window.innerWidth;
    const height = video.videoHeight || window.innerHeight;

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");

    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);

        const detector = await HandLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath: MODEL_URL,
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
          }
        );

        if (!cancelled) {
          detectorRef.current = detector;
          setReady(true);
        }
      } catch (e) {
        console.error(e);

        if (!cancelled) {
          setError(
            "MediaPipe yuklanmadi. Internet aloqasini tekshiring va sahifani qayta oching."
          );
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      detectorRef.current?.close();
    };
  }, []);

  useEffect(() => {
    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
    };
  }, [resizeCanvas]);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  const clearDrawing = useCallback(() => {
    strokesRef.current = [];
    particlesRef.current = [];
    lastPointRef.current = null;
    eraserPointRef.current = null;
    handLandmarksRef.current = null;
    drawingSessionRef.current = false;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  }, []);

  const addParticles = useCallback((x: number, y: number) => {
    const particles = particlesRef.current;

    for (let i = 0; i < 2; i++) {
      particles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 2.8,
        vy: (Math.random() - 0.5) * 2.8,
        life: 1,
        maxLife: 0.45 + Math.random() * 0.55,
        size: 1 + Math.random() * 2.5,
      });
    }

    if (particles.length > 400) {
      particles.splice(0, particles.length - 400);
    }
  }, []);

  /**
   * Remove the part of every stroke that is close to the
   * two-finger eraser cursor.
   *
   * We do not delete the whole stroke. Instead, each stroke is
   * split into smaller strokes around the erased area, giving the
   * natural "rubber" behavior from the reference.
   */
  const eraseAt = useCallback((x: number, y: number) => {
    const radius = 42;
    const radiusSq = radius * radius;
    const oldStrokes = strokesRef.current;
    const nextStrokes: Point[][] = [];

    for (const stroke of oldStrokes) {
      let segment: Point[] = [];

      for (const point of stroke) {
        const dx = point.x - x;
        const dy = point.y - y;

        if (dx * dx + dy * dy <= radiusSq) {
          if (segment.length > 1) {
            nextStrokes.push(segment);
          }
          segment = [];
          continue;
        }

        segment.push(point);
      }

      if (segment.length > 1) {
        nextStrokes.push(segment);
      }
    }

    strokesRef.current = nextStrokes;
  }, []);

  /**
   * Detects approximately:
   *   ☝️ = draw
   *   ✌️ = erase
   *
   * For erasing, index and middle fingers must be extended while
   * ring and pinky are folded. Thumb is ignored so the gesture is
   * comfortable to use from different angles.
   */
  function isTwoFingerErasePose(hand: NormalizedLandmark[]) {
    if (hand.length < 21) return false;

    const wrist = hand[0];

    const indexTip = hand[8];
    const indexPip = hand[6];

    const middleTip = hand[12];
    const middlePip = hand[10];

    const ringTip = hand[16];
    const ringPip = hand[14];

    const pinkyTip = hand[20];
    const pinkyPip = hand[18];

    const indexExtended =
      distance(indexTip, wrist) >
      distance(indexPip, wrist) * 1.08 &&
      angle(hand[5], hand[6], hand[7]) > 145 &&
      angle(hand[6], hand[7], hand[8]) > 135;

    const middleExtended =
      distance(middleTip, wrist) >
      distance(middlePip, wrist) * 1.08 &&
      angle(hand[9], hand[10], hand[11]) > 145 &&
      angle(hand[10], hand[11], hand[12]) > 135;

    const ringFolded =
      distance(ringTip, wrist) <
      distance(ringPip, wrist) * 1.10;

    const pinkyFolded =
      distance(pinkyTip, wrist) <
      distance(pinkyPip, wrist) * 1.10;

    return (
      indexExtended &&
      middleExtended &&
      ringFolded &&
      pinkyFolded
    );
  }

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    ctx.clearRect(0, 0, width, height);

    // LIVE NEON HAND: draw the detected 21-point hand as a wireframe.
    // The camera stays active for tracking, but its image is hidden.
    const hand = handLandmarksRef.current;
    const liveVideo = videoRef.current;

    if (
      hand &&
      hand.length >= 21 &&
      liveVideo &&
      liveVideo.videoWidth > 0 &&
      liveVideo.videoHeight > 0
    ) {
      const videoWidth = liveVideo.videoWidth;
      const videoHeight = liveVideo.videoHeight;
      const viewWidth = canvas.clientWidth;
      const viewHeight = canvas.clientHeight;

      const coverScale = Math.max(
        viewWidth / videoWidth,
        viewHeight / videoHeight
      );

      const renderedWidth = videoWidth * coverScale;
      const renderedHeight = videoHeight * coverScale;
      const cropX = (renderedWidth - viewWidth) / 2;
      const cropY = (renderedHeight - viewHeight) / 2;

      const pts = hand.map((lm) => ({
        x: (1 - lm.x) * renderedWidth - cropX,
        y: lm.y * renderedHeight - cropY,
      }));

      const connections: [number, number][] = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20],
        [5,9],[9,13],[13,17],
      ];

      ctx.save();
      ctx.beginPath();
      for (const [a,b] of connections) {
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
      }
      ctx.strokeStyle = "rgba(0, 234, 255, 0.32)";
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 25;
      ctx.shadowColor = "#00eaff";
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      for (const [a,b] of connections) {
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
      }
      ctx.strokeStyle = "#00eaff";
      ctx.lineWidth = 3.2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 12;
      ctx.shadowColor = "#00eaff";
      ctx.stroke();
      ctx.restore();

      ctx.save();
      for (const pt of pts) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.8, 0, Math.PI * 2);
        ctx.fillStyle = "#eaffff";
        ctx.shadowBlur = 14;
        ctx.shadowColor = "#00eaff";
        ctx.fill();
      }
      ctx.restore();

      ctx.save();
      for (const index of [4,8,12,16,20]) {
        const pt = pts[index];
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.shadowBlur = 22;
        ctx.shadowColor = "#00eaff";
        ctx.fill();
      }
      ctx.restore();
    }

    const strokes = strokesRef.current;

    // Every drawing session is rendered independently.
    // There is intentionally no path connecting two strokes.
    for (const stroke of strokes) {
      if (stroke.length < 2) continue;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);

      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y);
      }

      ctx.strokeStyle = "rgba(0, 220, 255, 0.35)";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 30;
      ctx.shadowColor = "#00e5ff";
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);

      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y);
      }

      ctx.strokeStyle = "#00eaff";
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 18;
      ctx.shadowColor = "#00eaff";
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);

      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y);
      }

      ctx.strokeStyle = "rgba(235, 255, 255, 0.9)";
      ctx.lineWidth = 1.15;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.restore();
    }
    // Finger cursor / eraser cursor.
    const last = erasing
      ? eraserPointRef.current
      : lastPointRef.current;

    if (last) {
      ctx.save();

      ctx.beginPath();
      ctx.arc(last.x, last.y, erasing ? 28 : 8, 0, Math.PI * 2);

      ctx.fillStyle = erasing
        ? "rgba(255, 255, 255, 0.10)"
        : "#eaffff";

      ctx.strokeStyle = erasing
        ? "rgba(255, 110, 110, 0.95)"
        : "rgba(0, 234, 255, 0.75)";

      ctx.lineWidth = erasing ? 2 : 1.5;
      ctx.shadowBlur = erasing ? 18 : 28;
      ctx.shadowColor = erasing ? "#ff5555" : "#00eaff";

      ctx.fill();
      ctx.stroke();

      if (!erasing) {
        ctx.beginPath();
        ctx.arc(last.x, last.y, 16, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 234, 255, 0.65)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.restore();
    }
    // Particles
    const particles = particlesRef.current;

    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];

      p.x += p.vx;
      p.y += p.vy;
      p.life -= 0.025;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      const alpha = Math.min(1, p.life / p.maxLife);

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = "#cfffff";
      ctx.shadowBlur = 12;
      ctx.shadowColor = "#00eaff";
      ctx.fill();
      ctx.restore();
    }
  }, [erasing, indexOnly]);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video || !detector || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const now = performance.now();

    if (lastTimeRef.current < 0 || now > lastTimeRef.current) {
      const result: HandLandmarkerResult =
        detector.detectForVideo(video, now);

      lastTimeRef.current = now;

      const canvas = canvasRef.current;

      if (canvas && result.landmarks.length > 0) {
        const hand = result.landmarks[0];
        handLandmarksRef.current = hand;

        const validPose = isIndexOnlyPose(hand);
        const erasePose = isTwoFingerErasePose(hand);

        setIndexOnly(validPose);
        setErasing(erasePose);

        const indexTip = hand[8];
        const middleTip = hand[12];

        // Convert MediaPipe camera coordinates to the exact visible
        // position of the mirrored object-fit: cover video.
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const viewWidth = canvas.clientWidth;
        const viewHeight = canvas.clientHeight;

        const coverScale = Math.max(
          viewWidth / videoWidth,
          viewHeight / videoHeight
        );

        const renderedWidth = videoWidth * coverScale;
        const renderedHeight = videoHeight * coverScale;

        const cropX = (renderedWidth - viewWidth) / 2;
        const cropY = (renderedHeight - viewHeight) / 2;

        const targetX =
          (1 - indexTip.x) * renderedWidth - cropX;

        const targetY =
          indexTip.y * renderedHeight - cropY;

        const middleX =
          (1 - middleTip.x) * renderedWidth - cropX;

        const middleY =
          middleTip.y * renderedHeight - cropY;

        const previous = lastPointRef.current;

        const x = previous
          ? previous.x * SMOOTHING +
            targetX * (1 - SMOOTHING)
          : targetX;

        const y = previous
          ? previous.y * SMOOTHING +
            targetY * (1 - SMOOTHING)
          : targetY;

        const point = {
          x,
          y,
          t: now,
        };

        lastPointRef.current = point;

        eraserPointRef.current = {
          x: (x + middleX) / 2,
          y: (y + middleY) / 2,
          t: now,
        };

        // THIS is the important part:
        // Draw ONLY when the hand pose is ☝️.
        if (erasePose) {
          // ✌️ = RUBBER / ERASER.
          // The eraser always ends the active drawing session.
          const eraserX = (targetX + middleX) / 2;
          const eraserY = (targetY + middleY) / 2;

          eraseAt(eraserX, eraserY);

          drawingSessionRef.current = false;
          lastPointRef.current = null;
        } else if (validPose) {
          const strokes = strokesRef.current;

          // CRITICAL:
          // Start a NEW stroke if the previous frame was not a valid
          // ☝️ drawing frame. Never use the old stroke's last point
          // as the first point of a new session.
          if (!drawingSessionRef.current) {
            strokes.push([point]);
            drawingSessionRef.current = true;
          } else {
            const currentStroke = strokes[strokes.length - 1];
            const last = currentStroke?.[currentStroke.length - 1];

            if (
              currentStroke &&
              (!last ||
                Math.hypot(x - last.x, y - last.y) > 1.2)
            ) {
              currentStroke.push(point);
            }
          }

          let totalPoints = strokes.reduce(
            (total, stroke) => total + stroke.length,
            0
          );

          while (totalPoints > MAX_POINTS && strokes.length > 1) {
            const removed = strokes.shift();
            totalPoints -= removed?.length ?? 0;
          }

          addParticles(x, y);
        } else {
          // ✊ / 🖐️ / any other pose = do nothing AND HARD-STOP
          // the current drawing session.
          drawingSessionRef.current = false;
          lastPointRef.current = null;
        }
      } else {
        lastPointRef.current = null;
        eraserPointRef.current = null;
        setIndexOnly(false);
        setErasing(false);
      }
    }

    render();

    rafRef.current = requestAnimationFrame(processFrame);
  }, [addParticles, render]);

  async function startCamera() {
    try {
      setError("");

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        });

      streamRef.current = stream;

      const video = videoRef.current;

      if (!video) return;

      video.srcObject = stream;

      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => resolve();
      });

      await video.play();

      resizeCanvas();

      setCameraOn(true);

      lastTimeRef.current = -1;

      rafRef.current =
        requestAnimationFrame(processFrame);
    } catch (e) {
      console.error(e);

      setError(
        "Kameraga ruxsat berilmadi yoki kamera topilmadi. Browser permission-ni tekshiring."
      );
    }
  }

  function stopCamera() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    const video = videoRef.current;

    if (video) {
      video.srcObject = null;
    }

    setCameraOn(false);
    setIndexOnly(false);

    lastPointRef.current = null;
    drawingSessionRef.current = false;
    handLandmarksRef.current = null;
    lastTimeRef.current = -1;
  }

  return (
    <main className="app">
      <video
        ref={videoRef}
        className="video"
        playsInline
        muted
      />

      <canvas
        ref={canvasRef}
        className="canvas"
      />

      <div className="topbar">
        <div>
          <div className="brand">NEON HAND</div>
          <div className="subtitle">
            INDEX FINGER DRAWING
          </div>
        </div>

        <div
          className={`status ${
            indexOnly ? "drawing-status" : ""
          }`}
        >
          <span
            className={`dot ${
              ready ? "ready" : ""
            } ${indexOnly ? "drawing-dot" : ""}`}
          />

          {!ready
            ? "LOADING MODEL..."
            : !cameraOn
              ? "CAMERA OFF"
              : erasing
                ? "✌ ERASER"
                : indexOnly
                  ? "☝ DRAWING"
                  : "✊ NOT DRAWING"}
        </div>
      </div>

      {!cameraOn && (
        <button
          className="start-button"
          disabled={!ready}
          onClick={startCamera}
        >
          {ready ? "START CAMERA" : "LOADING..."}
        </button>
      )}

      {cameraOn && (
        <div className="controls">
          <button
            className="secondary"
            onClick={clearDrawing}
          >
            CLEAR
          </button>

          <button
            className="secondary"
            onClick={stopCamera}
          >
            STOP CAMERA
          </button>
        </div>
      )}

      <div className="hint">
        {cameraOn
          ? erasing
            ? "✌ Ikki barmoq — RUBBER: neon chiziqni o‘chiradi"
            : indexOnly
              ? "☝ Ko‘rsatkich barmoq — chizadi"
              : "✊ Musht / boshqa holat — hech narsa qilmaydi"
          : "☝ chizadi  •  ✌ o‘chiradi  •  ✊ hech narsa qilmaydi"}
      </div>

      {error && (
        <div className="error">
          {error}
        </div>
      )}
    </main>
  );
}
