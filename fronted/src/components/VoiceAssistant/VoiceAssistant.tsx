import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppDispatch, useAppSelector } from "@/store";
import { fetchOverallScoring } from "@/store/slices/assistant/assistantSlice";
import { useRouter } from "next/navigation";

const getUsername = () => {
  return localStorage.getItem("username") || "unknown_user";
};

const getAuthToken = () => {
  if (typeof window === "undefined") return "";
  try {
    const rawToken =
      localStorage.getItem("access_token") ||
      sessionStorage.getItem("access_token") ||
      "";
    return rawToken ? `Bearer ${rawToken}` : "";
  } catch (error) {
    console.error("Error retrieving auth token:", error);
    return "";
  }
};

const CLASS_OPTIONS = [
  "Class 1",
  "Class 2",
  "Class 3",
  "Class 4",
  "Class 5",
  "Class 6",
  "Class 7",
  "Class 8",
  "Class 9",
  "Class 10",
];

const ACCENT_OPTIONS = [
  "American",
  "British",
  "Australian",
  "Indian",
  "Canadian",
];

const MOOD_OPTIONS = [
  "Neutral",
  "Happy",
  "Excited",
  "Calm",
  "Serious",
  "Playful",
];

export default function VoiceAssistant() {
  const dispatch = useAppDispatch();
  const router = useRouter();
  const [status, setStatus] = useState("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [transcription, setTranscription] = useState("");
  const localAudioRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const pcRef = useRef(null);
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const analyzerRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const rawAudioRef = useRef([]);
  const sendIntervalRef = useRef(null);
  const isPlayingRef = useRef(false);
  const audioQueueRef = useRef([]);
  const playbackContextRef = useRef(null); // New ref for playback context

  const [classOption, setClassOption] = useState("");
  const [accentOption, setAccentOption] = useState("");
  const [topicInput, setTopicInput] = useState("");
  const [moodOption, setMoodOption] = useState("");
  const [errors, setErrors] = useState({
    class: false,
    accent: false,
    topic: false,
    mood: false,
  });

  const scoringState = useAppSelector((state) => state.assistant);
  const [essayId, setEssayId] = useState("");
  const [showResultButton, setShowResultButton] = useState(false);
  const [loadingResult, setLoadingResult] = useState(false);
  const [loadingText, setLoadingText] = useState("Preparing your results...");

  const validateForm = () => {
    const newErrors = {
      class: !classOption,
      accent: !accentOption,
      topic: !topicInput.trim(),
      mood: !moodOption,
    };
    setErrors(newErrors);
    return !Object.values(newErrors).some((error) => error);
  };

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const initializeAudioProcessing = async (stream) => {
    // Create audio context if needed
    try {
      if (
        !audioContextRef.current ||
        audioContextRef.current.state === "closed"
      ) {
        const AudioContextClass =
          window.AudioContext || (window as any).webkitAudioContext;
        audioContextRef.current = new AudioContextClass({ sampleRate: 16000 });
      }

      const audioContext = audioContextRef.current;

      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      // Create analyzer for visualization
      const analyzer = audioContext.createAnalyser();
      analyzerRef.current = analyzer;
      analyzer.fftSize = 256;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyzer);

      // Create processor for audio capture
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Process audio chunks
      processor.onaudioprocess = (event) => {
        if (isPlayingRef.current) return;
        const inputData = event.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        rawAudioRef.current.push(pcmData);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      visualizeAudio();
    } catch (error) {
      console.error("Audio processing init error:", error);
      setStatus("error");
    }
  };

  const playAudioBuffer = async (arrayBuffer) => {
    if (isPlayingRef.current) {
      audioQueueRef.current.push(arrayBuffer);
      return;
    }

    try {
      setStatus("playing");
      isPlayingRef.current = true;

      // Mute microphone during playback
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => {
          track.enabled = false;
        });
      }

      // Create new context for playback
      playbackContextRef.current = new (window.AudioContext ||
        (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });

      const context = playbackContextRef.current;
      const audioBuffer = await context.decodeAudioData(arrayBuffer);
      const source = context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(context.destination);
      source.start(0);

      source.onended = () => {
        // Unmute microphone
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => {
            track.enabled = true;
          });
        }

        // Clean up playback context
        context.close();
        playbackContextRef.current = null;

        // Handle queue
        if (audioQueueRef.current.length > 0) {
          const nextBuffer = audioQueueRef.current.shift();
          playAudioBuffer(nextBuffer);
        } else {
          setStatus("connected");
          isPlayingRef.current = false;
          setTranscription("");
        }
      };
    } catch (e) {
      console.error("Playback error:", e);
      // Ensure mic is unmuted on error
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => {
          track.enabled = true;
        });
      }
      setStatus("connected");
      isPlayingRef.current = false;

      if (playbackContextRef.current) {
        playbackContextRef.current.close();
        playbackContextRef.current = null;
      }

      if (audioQueueRef.current.length > 0) {
        const nextBuffer = audioQueueRef.current.shift();
        playAudioBuffer(nextBuffer);
      } else {
        // FIX: Ensure status reset when queue is empty
        isPlayingRef.current = false;
      }
    }
  };

  const initWebRTC = async () => {
    setEssayId("");
    setShowResultButton(false);
    setLoadingResult(false);

    if (!validateForm()) {
      return;
    }
    try {
      setStatus("connecting");
      console.log("Getting user media...");

      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          noiseSuppression: true,
          echoCancellation: true,
        },
      });
      mediaStreamRef.current = stream;

      if (localAudioRef.current) {
        localAudioRef.current.srcObject = stream;
      }

      // Initialize audio processing
      await initializeAudioProcessing(stream);

      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          // Add your TURN servers here if needed
        ],
      });
      pcRef.current = pc;
      console.log("PeerConnection created");

      // Add local audio track
      stream.getTracks().forEach((track) => {
        pc.addTrack(track, stream);
      });

      // Monitor connection state
      pc.onconnectionstatechange = () => {
        console.log("Connection state:", pc.connectionState);
        if (
          pc.connectionState === "disconnected" ||
          pc.connectionState === "failed"
        ) {
          setStatus("error");
        }
      };

      pc.onicecandidate = (event) => {
        if (event.candidate) {
          const ws = wsRef.current;

          if (ws && ws.readyState === WebSocket.OPEN) {
            console.log("Sending ICE candidate", event.candidate);
            ws.send(
              JSON.stringify({
                type: "candidate",
                candidate: event.candidate,
              })
            );
          }
        }
      };

      pc.ontrack = (event) => {
        console.log("Received remote tracks");
        if (remoteAudioRef.current && !remoteAudioRef.current.srcObject) {
          remoteAudioRef.current.srcObject = event.streams[0];
          remoteAudioRef.current
            .play()
            .catch((e) => console.error("Remote audio play error:", e));
        }
      };

      const username = getUsername();
      const authHeader = getAuthToken();
      const token = authHeader.replace(/^Bearer\s+/i, "");

      const wsUrl = new URL("wss://llm.edusmartai.com/api/ws/assistant");
      wsUrl.searchParams.append("username", username);
      wsUrl.searchParams.append("token", token);
      wsUrl.searchParams.append("student_class", classOption);
      wsUrl.searchParams.append("accent", accentOption);
      wsUrl.searchParams.append("topic", topicInput);
      wsUrl.searchParams.append("mood", moodOption);

      const ws = new WebSocket(wsUrl.toString());
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");

        // Add ping mechanism to keep connection alive
        (ws as any).pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 2000);

        // Send audio chunks
        sendIntervalRef.current = setInterval(() => {
          if (
            rawAudioRef.current.length === 0 ||
            ws.readyState !== WebSocket.OPEN
          )
            return;

          const chunks = [...rawAudioRef.current];
          rawAudioRef.current = [];

          const totalLength = chunks.reduce(
            (acc, chunk) => acc + chunk.length,
            0
          );
          const combined = new Int16Array(totalLength);
          let offset = 0;

          chunks.forEach((chunk) => {
            combined.set(chunk, offset);
            offset += chunk.length;
          });

          ws.send(combined.buffer);
        }, 3000);

        // Create offer
        pc.createOffer({
          offerToReceiveAudio: true,
          offerToReceiveVideo: false,
        })
          .then((offer) => {
            console.log("Created offer:", offer);
            return pc.setLocalDescription(offer);
          })
          .then(() => {
            console.log("Sending offer:", pc.localDescription);
            ws.send(
              JSON.stringify({
                type: "offer",
                sdp: pc.localDescription.sdp,
              })
            );
          })
          .catch((error) => {
            console.error("Offer error:", error);
            setStatus("error");
          });
      };

      ws.onmessage = async (event) => {
        console.log("WebSocket message:", event.data);

        if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
          console.log("Received audio data");
          const arrayBuffer = await (event.data instanceof Blob
            ? event.data.arrayBuffer()
            : event.data);
          playAudioBuffer(arrayBuffer);
        }
        // Handle text messages
        else if (typeof event.data === "string") {
          console.log("111111111111111111111111");
          try {
            const message = JSON.parse(event.data);
            console.log("message", message);
            if (message.essay_id) {
              console.log("Received essay ID:", message.essay_id);
              setEssayId(message.essay_id);
            }
            if (message.type === "answer") {
              console.log("Received answer");
              await pc.setRemoteDescription(new RTCSessionDescription(message));
              setStatus("connected");
            } else if (message.type === "candidate") {
              console.log("Received ICE candidate");
              await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
            } else if (message.type === "transcript") {
              console.log("Received transcription:", message.text);
              setTranscription(message.text);
            }
          } catch (error) {
            console.error("Message handling error:", error);
          }
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus("error");
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        setStatus("idle");
        cleanup();
      };
    } catch (error) {
      console.error("WebRTC setup error:", error);
      setStatus("error");
      cleanup();
    }
  };

  const visualizeAudio = () => {
    if (!analyzerRef.current) return;

    const analyzer = analyzerRef.current;
    const bufferLength = analyzer.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyzer.getByteFrequencyData(dataArray);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const average = sum / bufferLength;
      setAudioLevel(average / 128);
    };

    draw();
  };

  const cleanup = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }

    if (wsRef.current) {
      // Clear ping interval
      if (wsRef.current.pingInterval) {
        clearInterval(wsRef.current.pingInterval);
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current
        .close()
        .catch((e) => console.error("Error closing audio context:", e));
      audioContextRef.current = null;
    }

    // Close playback context if exists
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }

    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    rawAudioRef.current = [];
    setTranscription("");
  };

  const stopAssistant = () => {
    setStatus("idle");
    cleanup();
    if (essayId) {
      setLoadingResult(true);
      setTimeout(() => {
        setLoadingResult(false);
        setShowResultButton(true);
      }, 3000);
    }
  };

  const handleShowResult = async () => {
    if (essayId) {
      try {
        setLoadingResult(true);
        setLoadingText("Preparing your results...");

        // Set timeout to update loading text after 15 seconds
        const textTimeout = setTimeout(() => {
          if (loadingResult) {
            setLoadingText("Almost there, finalizing results...");
          }
        }, 15000);

        // Fetch scoring data and wait
        await dispatch(fetchOverallScoring(essayId)).unwrap();

        // Clear timeout and navigate
        clearTimeout(textTimeout);
        router.push(`/assistantresult?essay_id=${essayId}`);
      } catch (error) {
        console.error("Failed to fetch scoring:", error);
        setLoadingResult(false);
      }
    }
  };

  const statusColors = {
    idle: "bg-gray-500",
    connecting: "bg-yellow-500",
    connected: "bg-green-500",
    playing: "bg-blue-500",
    error: "bg-red-500",
  };

  const statusMessages = {
    idle: "Ready to start",
    connecting: "Connecting...",
    connected: "Listening...",
    playing: "Assistant is speaking...",
    error: "Connection error",
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-5xl bg-white/90 backdrop-blur-lg rounded-2xl shadow-xl overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 text-center">
          <h1 className="text-2xl font-bold text-white">Speech Assistant</h1>
          <p className="text-indigo-200 mt-1">AI-powered voice assistant</p>
        </div>
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800 mb-3">
            Assistant Settings
          </h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Class {errors.class && <span className="text-red-600">*</span>}
              </label>
              <select
                value={classOption}
                onChange={(e) => {
                  setClassOption(e.target.value);
                  if (errors.class) setErrors({ ...errors, class: false });
                }}
                className={`w-full text-black rounded-lg border ${
                  errors.class ? "border-red-500" : "border-gray-300"
                } bg-white py-2 px-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent`}
              >
                {CLASS_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {errors.class && (
                <p className="mt-1 text-sm text-red-600">
                  Please select a class
                </p>
              )}
            </div>

            {/* Accent Dropdown */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Accent{" "}
                {errors.accent && <span className="text-red-600">*</span>}
              </label>
              <select
                value={accentOption}
                onChange={(e) => {
                  setAccentOption(e.target.value);
                  if (errors.accent) setErrors({ ...errors, accent: false });
                }}
                className={`w-full text-black rounded-lg border ${
                  errors.accent ? "border-red-500" : "border-gray-300"
                } bg-white py-2 px-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent`}
              >
                {ACCENT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {errors.accent && (
                <p className="mt-1 text-sm text-red-600">
                  Please select an accent
                </p>
              )}
            </div>

            {/* Topic Input */}
            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Topic {errors.topic && <span className="text-red-600">*</span>}
              </label>
              <input
                type="text"
                value={topicInput}
                onChange={(e) => {
                  setTopicInput(e.target.value);
                  if (errors.topic) setErrors({ ...errors, topic: false });
                }}
                placeholder="Enter discussion topic"
                className={`w-full text-black rounded-lg border ${
                  errors.topic ? "border-red-500" : "border-gray-300"
                } bg-white py-2 px-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent`}
              />
              {errors.topic && (
                <p className="mt-1 text-sm text-red-600">
                  Please enter a topic
                </p>
              )}
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Mood {errors.mood && <span className="text-red-600">*</span>}
              </label>
              <select
                value={moodOption}
                onChange={(e) => {
                  setMoodOption(e.target.value);
                  if (errors.mood) setErrors({ ...errors, mood: false });
                }}
                className={`w-full text-black rounded-lg border ${
                  errors.mood ? "border-red-500" : "border-gray-300"
                } bg-white py-2 px-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent`}
              >
                {MOOD_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              {errors.mood && (
                <p className="mt-1 text-sm text-red-600">
                  Please select a mood
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="p-8 flex flex-col items-center">
          <div className="relative mb-8">
            <div className="w-48 h-48 rounded-full bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center shadow-inner">
              <AnimatePresence>
                {status === "connected" && (
                  <motion.div
                    className="absolute inset-0 rounded-full bg-indigo-200 opacity-20"
                    animate={{ scale: [1, 1.3, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}
              </AnimatePresence>

              <div className="relative z-10">
                <div className="w-32 h-32 rounded-full bg-white shadow-lg flex items-center justify-center">
                  <motion.button
                    onClick={status === "idle" ? initWebRTC : stopAssistant}
                    className={`w-20 h-20 rounded-full flex items-center justify-center focus:outline-none ${
                      status === "idle"
                        ? "bg-indigo-100 hover:bg-indigo-200 text-indigo-600"
                        : "bg-red-500 hover:bg-red-600 text-white"
                    }`}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {status === "idle" ? (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-10 w-10"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                        />
                      </svg>
                    ) : (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-10 w-10"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    )}
                  </motion.button>
                </div>
              </div>
            </div>

            {(status === "connected" || status === "playing") && (
              <div className="absolute bottom-0 left-0 right-0 flex justify-center space-x-1 h-8">
                {Array.from({ length: 15 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className={`w-2 rounded-t ${
                      status === "playing" ? "bg-blue-500" : "bg-indigo-500"
                    }`}
                    animate={{
                      height: Math.max(
                        4,
                        audioLevel * 20 * (i % 3 === 0 ? 1.5 : 1)
                      ),
                    }}
                    transition={{
                      duration: 0.1,
                      ease: "easeOut",
                    }}
                    style={{ originY: 1 }}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Status Indicator */}
          <div className="flex flex-col items-center w-full">
            <div className="flex items-center mb-2">
              <motion.div
                className={`w-3 h-3 rounded-full mr-2 ${statusColors[status]}`}
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              />
              <span className="text-gray-700 font-medium">
                {statusMessages[status]}
              </span>
            </div>

            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${statusColors[status]}`}
                initial={{ width: 0 }}
                animate={{
                  width:
                    status === "connecting"
                      ? "60%"
                      : status === "connected"
                      ? "100%"
                      : status === "playing"
                      ? "100%"
                      : "0%",
                }}
                transition={{
                  duration: status === "connecting" ? 2 : 0.5,
                  repeat: status === "connecting" ? Infinity : 0,
                  repeatType: "reverse",
                }}
              />
            </div>
          </div>

          <div className="mt-8 text-center text-gray-600 text-sm">
            {status === "idle" && <p>Click the microphone to start speaking</p>}
            {status === "connected" && (
              <p>Speak naturally - I'm listening to you</p>
            )}
            {status === "playing" && (
              <p>Assistant is responding to your question</p>
            )}
          </div>
        </div>

        <div className="bg-gray-50 p-4 text-center text-gray-500 text-sm">
          Powered by WebRTC & AI
        </div>
        <div className="mt-4 p-4 bg-gray-100 rounded-lg max-w-full">
          <p className="text-sm text-gray-700 font-semibold">Transcription:</p>
          <p className="text-gray-600 break-words">{transcription}</p>
        </div>
      </div>
      <div className="mt-6 flex flex-col items-center">
        {/* Loading indicator */}
        {loadingResult && (
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-2"></div>
            <p className="text-gray-600 max-w-xs text-center">{loadingText}</p>
            <p className="text-xs text-gray-500 mt-2">
              This may take up to 15 seconds...
            </p>
          </div>
        )}

        {/* Result button */}
        {showResultButton && !loadingResult && (
          <motion.button
            onClick={handleShowResult}
            className="mt-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium py-3 px-8 rounded-full shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            View Detailed Results
          </motion.button>
        )}

        {/* Error message */}
        {scoringState.error && (
          <p className="mt-2 text-red-500 text-sm">
            Error loading results: {scoringState.error}
          </p>
        )}
      </div>
      <audio
        ref={localAudioRef}
        muted
        autoPlay
        playsInline
        className="hidden"
      />

      <audio ref={remoteAudioRef} autoPlay playsInline className="hidden" />
    </div>
  );
}
