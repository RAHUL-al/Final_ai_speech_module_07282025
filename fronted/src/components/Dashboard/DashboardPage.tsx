"use client";

import React, { useRef, useState, useEffect } from "react";
import { Formik, Form, Field, ErrorMessage } from "formik";
import * as Yup from "yup";
import { useAppDispatch, useAppSelector } from "../../store";
import { generateParagraph } from "../../store/slices/paragraphSlice";
import {
  FaPause,
  FaSpinner,
  FaVolumeUp,
  FaMagic,
  FaRobot,
  FaPlay,
} from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { IoIosArrowDown } from "react-icons/io";

const Dashboard = () => {
  const dispatch = useAppDispatch();
  const router = useRouter();
  const { generatedParagraph, isLoading, essayId } = useAppSelector(
    (state) => state.paragraph
  );

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeTab, setActiveTab] = useState("form"); // "form" or "paragraph"

  const getUsername = () => {
    return localStorage.getItem("username") || "unknown_user";
  };

  const getAuthToken = () => {
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

  const user = getUsername();

  // Options for dropdowns
  const classOptions = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
  ];

  const accentOptions = [
    "American",
    "British",
    "Australian",
    "Indian",
    "Canadian",
    "Irish",
  ];

  const moodOptions = [
    "Happy",
    "Excited",
    "Calm",
    "Serious",
    "Playful",
    "Confident",
  ];

  const difficultyOptions = ["Easy", "Medium", "Strong"];

  const initialValues = {
    class: "",
    accent: "",
    topic: "",
    mood: "",
    difficulty: "",
  };

  const validationSchema = Yup.object({
    class: Yup.string().required("Class is required"),
    accent: Yup.string().required("Accent is required"),
    topic: Yup.string()
      .required("Topic is required")
      .min(3, "Topic must be at least 3 characters"),
    mood: Yup.string().required("Mood is required"),
    difficulty: Yup.string().required("Difficulty is required"),
  });

  const handleSubmit = (values) => {
    dispatch(generateParagraph(values))
      .unwrap()
      .then(() => {
        setActiveTab("paragraph");
        toast.success("Paragraph generated successfully!");
      })
      .catch((error) => {
        toast.error(error || "Failed to generate paragraph");
      });
  };

  const handlePractice = () => {
    if (!generatedParagraph) {
      toast.error("No paragraph generated to practice");
      return;
    }

    // Encode and pass paragraph via URL
    const encodedParagraph = encodeURIComponent(generatedParagraph);
    router.push(`/practice?paragraph=${encodedParagraph}&essay_id=${essayId}`);
  };

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  const token = getAuthToken();

  const fetchAndPlayAudio = async () => {
    if (!user || !token) {
      toast.error("Authentication required");
      return;
    }

    setIsAudioLoading(true);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "api/";
      const username = user;
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
      const audioUrl = URL.createObjectURL(audioBlob);

      // Clean up previous audio if exists
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      // Create new audio instance
      audioRef.current = new Audio(audioUrl);
      audioRef.current.onended = () => setIsSpeaking(false);
      audioRef.current.onpause = () => setIsSpeaking(false);

      await audioRef.current.play();
      setIsSpeaking(true);
    } catch (error) {
      toast.error("Failed to play audio: " + error.message);
    } finally {
      setIsAudioLoading(false);
    }
  };

  const handleSpeakButtonClick = () => {
    if (isSpeaking && audioRef.current) {
      audioRef.current.pause();
      setIsSpeaking(false);
    } else {
      fetchAndPlayAudio();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center px-4 py-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-3xl bg-white shadow-2xl rounded-2xl overflow-hidden border border-gray-200"
      >
        {/* Gradient Header with AI Icon */}
        <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 py-5 px-6 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent to-white opacity-10"></div>
          <div className="absolute -right-10 -top-10 w-40 h-40 rounded-full bg-pink-500 opacity-20"></div>
          <div className="absolute -right-5 -bottom-10 w-32 h-32 rounded-full bg-indigo-500 opacity-20"></div>

          <div className="relative z-10 flex items-center justify-center gap-3">
            <motion.div
              animate={{ rotate: [0, 15, 0, -15, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <FaRobot className="text-white text-3xl" />
            </motion.div>
            <h2 className="text-2xl font-bold text-white text-center">
              AI English Practice Generator
            </h2>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab("form")}
            className={`flex-1 py-4 font-medium text-center transition-all duration-300 ${
              activeTab === "form"
                ? "text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50/50"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <FaMagic className="text-sm" />
              <span>Create Practice</span>
            </div>
          </button>
          <button
            onClick={() => generatedParagraph && setActiveTab("paragraph")}
            className={`flex-1 py-4 font-medium text-center transition-all duration-300 ${
              activeTab === "paragraph"
                ? "text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50/50"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-50" +
                  (generatedParagraph ? "" : " opacity-50 cursor-not-allowed")
            }`}
            disabled={!generatedParagraph}
          >
            <div className="flex items-center justify-center gap-2">
              <FaPlay className="text-sm" />
              <span>Your Practice</span>
            </div>
          </button>
        </div>

        <div className="p-6">
          <AnimatePresence mode="wait">
            {activeTab === "form" ? (
              <motion.div
                key="form"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <Formik
                  initialValues={initialValues}
                  validationSchema={validationSchema}
                  onSubmit={handleSubmit}
                >
                  {({ isValid, dirty }) => (
                    <Form>
                      <div className="grid gap-6 mb-6 md:grid-cols-2">
                        {/* Class Dropdown */}
                        <div>
                          <label className="block mb-2 text-sm font-medium text-gray-700">
                            Class <span className="text-red-500">*</span>
                          </label>
                          <div className="relative">
                            <Field
                              as="select"
                              name="class"
                              className="appearance-none bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block w-full p-3 pr-10 shadow-sm transition duration-200"
                            >
                              <option value="">Select class</option>
                              {classOptions.map((option) => (
                                <option key={option} value={option}>
                                  Grade {option}
                                </option>
                              ))}
                            </Field>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                              <IoIosArrowDown />
                            </div>
                          </div>
                          <ErrorMessage
                            name="class"
                            component="div"
                            className="text-red-500 text-sm mt-1"
                          />
                        </div>

                        {/* Accent Dropdown */}
                        <div>
                          <label className="block mb-2 text-sm font-medium text-gray-700">
                            Accent <span className="text-red-500">*</span>
                          </label>
                          <div className="relative">
                            <Field
                              as="select"
                              name="accent"
                              className="appearance-none bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block w-full p-3 pr-10 shadow-sm transition duration-200"
                            >
                              <option value="">Select accent</option>
                              {accentOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </Field>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                              <IoIosArrowDown />
                            </div>
                          </div>
                          <ErrorMessage
                            name="accent"
                            component="div"
                            className="text-red-500 text-sm mt-1"
                          />
                        </div>

                        {/* Topic Input */}
                        <div className="md:col-span-2">
                          <label className="block mb-2 text-sm font-medium text-gray-700">
                            Topic <span className="text-red-500">*</span>
                          </label>
                          <Field
                            type="text"
                            name="topic"
                            className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block w-full p-3 shadow-sm transition duration-200"
                            placeholder="What do you want to talk about?"
                          />
                          <ErrorMessage
                            name="topic"
                            component="div"
                            className="text-red-500 text-sm mt-1"
                          />
                        </div>

                        {/* Mood Dropdown */}
                        <div>
                          <label className="block mb-2 text-sm font-medium text-gray-700">
                            Mood <span className="text-red-500">*</span>
                          </label>
                          <div className="relative">
                            <Field
                              as="select"
                              name="mood"
                              className="appearance-none bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block w-full p-3 pr-10 shadow-sm transition duration-200"
                            >
                              <option value="">Select mood</option>
                              {moodOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </Field>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                              <IoIosArrowDown />
                            </div>
                          </div>
                          <ErrorMessage
                            name="mood"
                            component="div"
                            className="text-red-500 text-sm mt-1"
                          />
                        </div>

                        {/* Difficulty Dropdown */}
                        <div>
                          <label className="block mb-2 text-sm font-medium text-gray-700">
                            Difficulty Level{" "}
                            <span className="text-red-500">*</span>
                          </label>
                          <div className="relative">
                            <Field
                              as="select"
                              name="difficulty"
                              className="appearance-none bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block w-full p-3 pr-10 shadow-sm transition duration-200"
                            >
                              <option value="">Select difficulty</option>
                              {difficultyOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </Field>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                              <IoIosArrowDown />
                            </div>
                          </div>
                          <ErrorMessage
                            name="difficulty"
                            component="div"
                            className="text-red-500 text-sm mt-1"
                          />
                        </div>
                      </div>

                      {/* Submit Button */}
                      <div className="flex justify-center mt-8">
                        <motion.button
                          type="submit"
                          disabled={isLoading || !(isValid && dirty)}
                          className={`px-8 py-4 rounded-xl font-bold text-white shadow-lg transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-indigo-300/50 ${
                            isLoading || !(isValid && dirty)
                              ? "bg-gray-400 cursor-not-allowed"
                              : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                          }`}
                          whileHover={
                            !(isLoading || !(isValid && dirty))
                              ? {
                                  scale: 1.03,
                                  boxShadow:
                                    "0 10px 25px -5px rgba(99, 102, 241, 0.4)",
                                }
                              : {}
                          }
                          whileTap={
                            !(isLoading || !(isValid && dirty))
                              ? { scale: 0.98 }
                              : {}
                          }
                        >
                          {isLoading ? (
                            <span className="flex items-center">
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{
                                  duration: 1,
                                  repeat: Infinity,
                                  ease: "linear",
                                }}
                                className="mr-3"
                              >
                                <FaSpinner />
                              </motion.div>
                              Generating...
                            </span>
                          ) : (
                            <span className="flex items-center">
                              <FaMagic className="mr-2" />
                              Generate Paragraph
                            </span>
                          )}
                        </motion.button>
                      </div>
                    </Form>
                  )}
                </Formik>
              </motion.div>
            ) : (
              <motion.div
                key="paragraph"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
                className="mt-4"
              >
                {/* Generated Paragraph Section */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="p-6 bg-gradient-to-br from-indigo-50/50 to-purple-50/50 rounded-xl border border-indigo-100 shadow-sm"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-xl font-bold text-indigo-900">
                      Your AI-Generated Paragraph
                    </h3>

                    <motion.button
                      onClick={handleSpeakButtonClick}
                      disabled={isAudioLoading}
                      className={`p-3 rounded-xl relative overflow-hidden ${
                        isAudioLoading
                          ? "text-indigo-300 cursor-not-allowed"
                          : isSpeaking
                          ? "bg-indigo-100 text-indigo-700"
                          : "bg-white text-indigo-600 hover:bg-indigo-50 border border-indigo-200"
                      } transition-all duration-300 shadow-sm`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      aria-label={
                        isSpeaking ? "Pause audio" : "Listen to paragraph"
                      }
                    >
                      {isAudioLoading ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            ease: "linear",
                          }}
                        >
                          <FaSpinner className="text-xl" />
                        </motion.div>
                      ) : isSpeaking ? (
                        <FaPause className="text-xl" />
                      ) : (
                        <FaVolumeUp className="text-xl" />
                      )}

                      {/* Wave animation */}
                      {isSpeaking && !isAudioLoading && (
                        <div className="absolute inset-0 flex items-center justify-center -space-x-1">
                          {[...Array(5)].map((_, i) => (
                            <motion.span
                              key={i}
                              className="w-1 h-2 bg-indigo-500 rounded-full"
                              animate={{
                                height: ["0.25rem", "1rem", "0.25rem"],
                              }}
                              transition={{
                                duration: 1.5,
                                repeat: Infinity,
                                delay: i * 0.15,
                              }}
                            />
                          ))}
                        </div>
                      )}
                    </motion.button>
                  </div>

                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white p-6 rounded-xl shadow-inner border border-gray-100"
                  >
                    <p className="text-gray-700 leading-relaxed whitespace-pre-line text-lg">
                      {generatedParagraph}
                    </p>
                  </motion.div>

                  <div className="mt-6 flex justify-between">
                    <motion.button
                      onClick={() => setActiveTab("form")}
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.98 }}
                      className="px-5 py-2.5 text-indigo-600 bg-white border border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors font-medium shadow-sm"
                    >
                      Create New
                    </motion.button>

                    <motion.button
                      onClick={handlePractice}
                      whileHover={{
                        scale: 1.05,
                        boxShadow: "0 5px 15px -3px rgba(99, 102, 241, 0.3)",
                      }}
                      whileTap={{ scale: 0.98 }}
                      className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all font-bold shadow-md"
                    >
                      Start Practice Now
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
