"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { useAppDispatch, useAppSelector } from "../../store";
import {
  initializePractice,
  startRecording,
  stopRecording,
  resetPracticeState,
  setAnalysisResults,
} from "@/store/slices/practiceSlice";
import { useRouter, useSearchParams } from "next/navigation";
import {
  FaMicrophone,
  FaStop,
  FaPlay,
  FaArrowLeft,
  FaSpinner,
} from "react-icons/fa";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { FaPause, FaVolumeUp, FaVolumeMute } from "react-icons/fa";
import { fetchOverallScoring } from "@/store/slices/practiceSlice";

const PracticePage = () => {
  const dispatch = useAppDispatch();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [showResultButton, setShowResultButton] = useState(false);
  const essayId = searchParams.get("essay_id");
  const essay_id = searchParams.get("essay_id");

  const paragraph = searchParams.get("paragraph")
    ? decodeURIComponent(searchParams.get("paragraph")!)
    : null;

  const [debugAudioUrl, setDebugAudioUrl] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const monitorRef = useRef(null);

  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const rawAudioRef = useRef([]);

  const [showInfoNotice, setShowInfoNotice] = useState(true);

  const getUsername = () => {
    return localStorage.getItem("username") || "unknown_user";
  };

  const mediaStreamRef = useRef(null);

  const socketRef = useRef(null);

  const { isRecording, isAnalyzing, analysisResults, error } = useAppSelector(
    (state) => state.practice
  );

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

  const fetchAndPlayAudio = async () => {
    const token = getAuthToken();
    const username = getUsername();

    if (!username || !token) {
      toast.error("Authentication required");
      return;
    }

    setIsAudioLoading(true);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "api/";
      const response = await fetch(
        `${baseUrl}/get-tts-audio?username=${username}`,
        {
          headers: {
            Authorization: `${token}`,
          },
        }
      );

      if (!response.ok) throw new Error("Audio fetch failed");

      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      setAudioUrl(url);
      return url;
    } catch (error) {
      toast.error("Failed to load audio: " + error.message);
      return null;
    } finally {
      setIsAudioLoading(false);
    }
  };

  const toggleAudioPlayback = async () => {
    if (!audioRef.current) return;

    if (isAudioPlaying) {
      audioRef.current.pause();
      setIsAudioPlaying(false);
    } else {
      try {
        let url = audioUrl;
        if (!url) {
          url = await fetchAndPlayAudio();
        }

        if (url && audioRef.current) {
          if (audioRef.current.src !== url) {
            audioRef.current.src = url;
          }

          await audioRef.current.play();
          setIsAudioPlaying(true);
        }
      } catch (error) {
        console.error("Audio playback failed:", error);
        toast.error("Failed to play audio");
      }
    }
  };

  const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

  if (!BASE_URL) {
    console.error("❌ NEXT_PUBLIC_API_URL is not defined");
    toast.error("Configuration error: API URL missing");
    return;
  }

  const testWebSocketConnection = useCallback(async () => {
    const username = getUsername();
    if (typeof window === "undefined") return;
    const token = getAuthToken();
    console.log("Testing WebSocket connection with token:", token);
    if (!token) {
      toast.error("Authentication token missing");
      return;
    }

    const cleanBaseUrl = BASE_URL.replace(/^https?:\/\//, "");
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

    const params = new URLSearchParams({
      username: encodeURIComponent(username),
      token: encodeURIComponent(token),
    });
    if (essayId) {
      params.append("essay_id", essayId);
    }

    const testSocket = new WebSocket(
      `${protocol}//${cleanBaseUrl}/ws/audio?${params.toString()}`
    );
    testSocket.onopen = () => {
      console.log("[TEST] WebSocket connected");
      testSocket.send(
        JSON.stringify({ type: "test", data: "Connection test" })
      );
      testSocket.close();
    };

    testSocket.onerror = (error) => {
      console.error("[TEST] WebSocket error:", error);
    };

    testSocket.onmessage = (e) => {
      console.log("[TEST] Received:", e.data);
    };
  }, [BASE_URL, essayId]);

  const startMicrophoneMonitoring = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (monitorRef.current) {
        monitorRef.current.srcObject = stream;
        monitorRef.current
          .play()
          .catch((e) => console.error("Monitoring play error:", e));
        setIsMonitoring(true);
      }
    } catch (error) {
      console.error("Monitoring error:", error);
      toast.error("Microphone access failed during monitoring");
    }
  };

  const stopMicrophoneMonitoring = () => {
    if (monitorRef.current && monitorRef.current.srcObject) {
      monitorRef.current.srcObject.getTracks().forEach((track) => track.stop());
      monitorRef.current.srcObject = null;
      setIsMonitoring(false);
    }
  };

  const checkMicrophoneAccess = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter((d) => d.kind === "audioinput");

      if (mics.length === 0) {
        toast.error("No microphones detected");
        return false;
      }

      console.log("Available microphones:", mics);
      return true;
    } catch (error) {
      console.error("Device enumeration error:", error);
      toast.error("Couldn't access device list");
      return false;
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (paragraph) {
      dispatch(initializePractice(paragraph));
      testWebSocketConnection();
    } else {
      toast.error("No paragraph found for practice");
      router.push("/dashboard");
    }
  }, [dispatch, paragraph, router, testWebSocketConnection]);

  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const handleStartRecording = async () => {
    setShowInfoNotice(false);
    setDebugAudioUrl(null);
    rawAudioRef.current = [];

    const hasMicrophone = await checkMicrophoneAccess();
    if (!hasMicrophone) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          noiseSuppression: true,
          echoCancellation: true,
        },
      });
      const audioTracks = stream.getAudioTracks();
      console.log("Active audio tracks:", audioTracks);

      audioTracks.forEach((track) => {
        console.log("Track settings:", track.getSettings());
        console.log("Track enabled:", track.enabled);
        console.log("Track readyState:", track.readyState);
      });

      mediaStreamRef.current = stream;

      const AudioContextClass =
        window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AudioContextClass({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
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

      const token = getAuthToken();
      if (!token) {
        toast.error("Missing authentication token");
        return;
      }
      const username = getUsername();
      console.log("Using username:", username);

      const cleanBaseUrl = BASE_URL.replace(/^https?:\/\//, "");

      const protocol = window.location.protocol === "https:" ? "wss:" : "wss:";
      const params = new URLSearchParams({
        username: encodeURIComponent(username),
        token: encodeURIComponent(token),
      });
      if (essayId) {
        params.append("essay_id", essayId);
      }

      const socket = new WebSocket(
        `${protocol}//${cleanBaseUrl}/ws/audio?${params.toString()}`
      );
      console.log("Connecting to:", socket.url);
      socketRef.current = socket;

      // Handle socket events
      socket.onopen = () => {
        console.log("WebSocket connected");
        console.log("Protocol:", socket.protocol);
        console.log("Extensions:", socket.extensions);

        dispatch(startRecording());
        let sendInterval: NodeJS.Timeout;

        sendInterval = setInterval(() => {
          if (
            rawAudioRef.current.length === 0 ||
            socket.readyState !== WebSocket.OPEN
          )
            return;

          const chunks = rawAudioRef.current;
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

          socket.send(combined.buffer);
        }, 3000);
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("Analysis received:", data);
        dispatch(setAnalysisResults(data));
      };

      socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        toast.error("WebSocket connection error");
        dispatch(stopRecording());
      };

      socket.onclose = (event) => {
        console.error("WebSocket closed unexpectedly", {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
        });
      };

      const reader = new FileReader();

      // Send to WebSocket
      reader.onload = () => {
        if (socket.readyState === WebSocket.OPEN) {
          console.log("Sending audio chunk as ArrayBuffer");
          socket.send(reader.result);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    } catch (err) {
      console.error("Recording error:", err);
      toast.error("Failed to access microphone. Please check permissions.");
      dispatch(stopRecording());
    }
  };

  const handleStopRecording = () => {
    setShowInfoNotice(false);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }

    if (socketRef.current) {
      const socket = socketRef.current;

      if (
        rawAudioRef.current.length > 0 &&
        socket.readyState === WebSocket.OPEN
      ) {
        const chunks = rawAudioRef.current;
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

        socket.send(combined.buffer);
      }

      if (socket.sendInterval) {
        clearInterval(socket.sendInterval);
      }

      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ action: "end" }));
        socket.close(1000, "Recording completed");
      } else {
        socket.close();
      }
    }

    dispatch(stopRecording());
    setShowResultButton(true);
  };

  const handleBackToDashboard = () => {
    if (audioRef.current && isAudioPlaying) {
      audioRef.current.pause();
    }
    dispatch(resetPracticeState());
    router.push("/dashboard");
  };

  const handleSeeResults = () => {
    if (!essayId) {
      toast.error("Essay ID is missing");
      return;
    }

    // Set loading state
    dispatch(setAnalysisResults({ isAnalyzing: true })); // Update to accept object

    dispatch(fetchOverallScoring(essayId))
      .unwrap()
      .then((response) => {
        // Store results before navigation
        dispatch(
          setAnalysisResults({
            results: response,
            isAnalyzing: false,
          })
        );
        router.push(`/result?essayId=${essayId}`);
      })
      .catch((error) => {
        toast.error("Failed to fetch results: " + error.message);
        dispatch(setAnalysisResults({ isAnalyzing: false }));
      });
  };

  useEffect(() => {
    const audioEl = audioRef.current;
    return () => {
      if (audioEl) {
        audioEl.pause();
      }

      if (socketRef.current) socketRef.current.close();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [dispatch]);

  useEffect(() => {
    return () => {
      if (debugAudioUrl) URL.revokeObjectURL(debugAudioUrl);
    };
  }, [debugAudioUrl]);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-3xl bg-white shadow-xl rounded-xl overflow-hidden"
      >
        <div className="bg-indigo-600 py-4 px-6 flex items-center">
          <button
            onClick={handleBackToDashboard}
            className="text-white mr-4 p-2 rounded-full hover:bg-indigo-700 transition-colors"
          >
            <FaArrowLeft />
          </button>
          <h2 className="text-2xl font-bold text-white text-center flex-grow">
            Practice Your English
          </h2>
        </div>

        <div className="p-6">
          {isAnalyzing && !paragraph ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
              <span className="ml-4 text-gray-700">
                Loading practice session...
              </span>
            </div>
          ) : (
            <>
              {!isRecording && (
                <div className="mb-8 bg-indigo-50 p-4 rounded-xl border border-indigo-100 transition-all duration-500">
                  <h3 className="text-lg font-semibold text-indigo-800 mb-3">
                    Read this paragraph aloud:
                  </h3>
                  <div className="bg-white p-4 rounded-lg shadow-inner max-h-60 overflow-y-auto">
                    <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                      {paragraph}
                    </p>
                  </div>
                </div>
              )}

              <div className="mb-8 flex justify-center">
                <button
                  onClick={toggleAudioPlayback}
                  disabled={isAudioLoading || isRecording}
                  className={`flex items-center justify-center ${
                    isAudioLoading
                      ? "bg-gray-300 cursor-not-allowed"
                      : "bg-indigo-100 hover:bg-indigo-200"
                  } text-indigo-600 p-4 rounded-full transition-colors min-w-[180px]`}
                >
                  {isAudioLoading ? (
                    <FaSpinner className="animate-spin text-xl" />
                  ) : isAudioPlaying ? (
                    <FaPause className="text-xl" />
                  ) : (
                    <FaPlay className="text-xl" />
                  )}
                  <span className="ml-2">
                    {isAudioLoading
                      ? "Loading..."
                      : isAudioPlaying
                      ? "Pause"
                      : "Listen Again"}
                  </span>
                </button>

                <audio
                  ref={audioRef}
                  src={audioUrl}
                  onPlay={() => setIsAudioPlaying(true)}
                  onPause={() => setIsAudioPlaying(false)}
                  onEnded={() => setIsAudioPlaying(false)}
                  onLoadedData={() => setIsAudioLoading(false)}
                  onWaiting={() => setIsAudioLoading(true)}
                  onCanPlay={() => setIsAudioLoading(false)}
                  className="hidden"
                />
              </div>

              <div className="flex flex-col items-center justify-center my-8">
                {showInfoNotice && !isRecording && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded"
                  >
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg
                          className="h-5 w-5 text-yellow-400"
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm text-yellow-700">
                          The paragraph will be hidden when you start recording
                          to encourage memorization and natural speech.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {!isRecording ? (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleStartRecording}
                    className="bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-full p-6 shadow-lg hover:shadow-xl transition-all duration-300"
                    disabled={isAnalyzing}
                  >
                    <FaMicrophone className="text-3xl" />
                  </motion.button>
                ) : (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleStopRecording}
                    className="bg-gradient-to-r from-red-500 to-orange-500 text-white rounded-full p-6 shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <FaStop className="text-3xl" />
                  </motion.button>
                )}

                <p className="mt-4 text-gray-600">
                  {isRecording
                    ? "Recording... Click to stop."
                    : "Click the microphone to start recording."}
                </p>

                {isRecording && (
                  <div className="mt-4 flex space-x-2">
                    {[...Array(5)].map((_, i) => (
                      <motion.div
                        key={i}
                        animate={{ height: [10, 30, 10] }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                        className="w-3 bg-indigo-500 rounded-full"
                      />
                    ))}
                  </div>
                )}
                {showResultButton && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-8 flex justify-center"
                  >
                    <button
                      onClick={handleSeeResults}
                      disabled={isAnalyzing}
                      className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-8 py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 text-lg font-semibold flex items-center disabled:opacity-75"
                    >
                      {isAnalyzing ? (
                        <FaSpinner className="animate-spin mr-2" />
                      ) : null}
                      See Detailed Results
                      {!isAnalyzing && (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-5 w-5 ml-2"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </button>
                  </motion.div>
                )}
              </div>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default PracticePage;
