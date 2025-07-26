// app/result/page.tsx
"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  FaArrowLeft,
  FaSpinner,
  FaChartLine,
  FaVolumeUp,
  FaLightbulb,
  FaRegSmile,
} from "react-icons/fa";
import { motion } from "framer-motion";

type ResultAPI = {
  understanding: string;
  topic_grip: string;
  suggestions: string[];
  pronunciation: number | string;
  grammar: number | string;
  fluency: number | string;
  emotion: string;
};

const ResultPage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const essayId = searchParams.get("essayId");
  const [resultData, setResultData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 10; // Max 10 retries (30 seconds total)
  const retryDelay = 3000; // 3 seconds between retries
  const fetchResults = useCallback(async () => {
    if (!essayId) return;

    try {
      const token =
        localStorage.getItem("access_token") ||
        sessionStorage.getItem("access_token") ||
        "";
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/overall-scoring-by-id?essay_id=${essayId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      if (!response.ok) throw new Error("Failed to fetch results");

      const data = await response.json();

      // Check if analysis is complete
      if (data && data.understanding) {
        setResultData(data);
        setLoading(false);
      } else if (retryCount < maxRetries) {
        // Retry after delay if data not ready
        setTimeout(() => {
          setRetryCount((prev) => prev + 1);
          fetchResults();
        }, retryDelay);
      } else {
        throw new Error("Analysis is taking too long. Please try again later.");
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }, [essayId, retryCount]);

  useEffect(() => {
    if (essayId) fetchResults();
    else {
      setError("Missing essay ID");
      setLoading(false);
    }
  }, [essayId, fetchResults]);

  const parseScore = (value) => {
    if (typeof value === "number") return value;
    if (typeof value === "string") {
      const num = parseFloat(value);
      return isNaN(num) ? 0 : num;
    }
    return 0;
  };

  const calculateOverallScore = () => {
    if (!resultData) return 0;

    const validScores = [
      parseScore(resultData.grammar),
      parseScore(resultData.pronunciation),
      parseScore(resultData.fluency),
    ].filter((score) => score > 0);

    if (validScores.length === 0) return 0;
    return (
      validScores.reduce((a, b) => a + b, 0) / validScores.length
    ).toFixed(1);
  };

  const result = resultData
    ? {
        overallScore: calculateOverallScore(),
        pronunciation: parseScore(resultData.pronunciation),
        fluency: parseScore(resultData.fluency),
        grammar: parseScore(resultData.grammar),
        emotion: resultData.emotion,
        understanding: resultData.understanding,
        topicGrip: resultData.topic_grip,
        suggestions: resultData.suggestions || [],
      }
    : {
        overallScore: 0,
        pronunciation: 0,
        fluency: 0,
        grammar: 0,
        emotion: "",
        understanding: "",
        topicGrip: "",
        suggestions: [],
      };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <FaSpinner className="animate-spin text-indigo-600 text-4xl mx-auto mb-4" />
          <p className="text-gray-700 text-lg">
            Analyzing your performance...{" "}
            {retryCount > 0 && "(This may take a moment)"}
          </p>
          {retryCount > 3 && (
            <p className="text-gray-500 mt-2">
              Still processing... {maxRetries - retryCount} attempts remaining
            </p>
          )}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="bg-white p-8 rounded-xl shadow-lg max-w-md text-center">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Error Loading Results
          </h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => router.push("/dashboard")}
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const emotion = resultData!.emotion;

  const breakdown = [
    { label: "Pronunciation", score: parseScore(resultData!.pronunciation) },
    { label: "Fluency", score: parseScore(resultData!.fluency) },
    { label: "Grammar", score: parseScore(resultData!.grammar) },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={() => router.back()}
            className="flex items-center text-indigo-600 hover:text-indigo-800 transition-colors"
          >
            <FaArrowLeft className="mr-2" />
            Back
          </button>
          <h1 className="text-3xl font-bold text-gray-800 text-center">
            Your Speaking Results
          </h1>
          <div className="w-10"></div> {/* Spacer for alignment */}
        </div>

        {/* Overall Score */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-xl overflow-hidden mb-8"
        >
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 text-white">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-semibold">Overall Score</h2>
                <p className="opacity-80">Based on your recording</p>
              </div>
              <div className="relative">
                <div className="w-28 h-28 rounded-full border-4 border-white/30 flex items-center justify-center">
                  <span className="text-4xl font-bold">
                    {result.overallScore}
                  </span>
                  <span className="text-lg">/10</span>
                </div>
                <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-white animate-spin-slow"></div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Score Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8"
        >
          <div className="bg-white p-6 rounded-2xl shadow-lg">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FaChartLine className="text-indigo-500 mr-2" />
              Score Breakdown
            </h3>
            <div className="space-y-4">
              {[
                { label: "Pronunciation", score: result.pronunciation },
                { label: "Fluency", score: result.fluency },
                { label: "Grammar", score: result.grammar },
              ].map((item, index) => (
                <div key={index}>
                  <div className="flex justify-between mb-1">
                    <span className="text-gray-700">{item.label}</span>
                    <span className="font-semibold">{item.score}/10</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-gradient-to-r from-indigo-400 to-indigo-600 h-2.5 rounded-full"
                      style={{ width: `${item.score * 10}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Emotion & Suggestions */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-lg">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FaRegSmile className="text-indigo-500 mr-2" />
                Emotion
              </h3>
              <div className="flex items-center">
                <span className="inline-block bg-indigo-100 text-indigo-800 text-lg px-4 py-2 rounded-full">
                  {result.emotion}
                </span>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-lg">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FaLightbulb className="text-yellow-500 mr-2" />
                Suggestions for Improvement
              </h3>
              <ul className="space-y-2">
                {result.suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start">
                    <div className="bg-yellow-100 p-1 rounded-full mr-3 mt-1">
                      <svg
                        className="w-4 h-4 text-yellow-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        ></path>
                      </svg>
                    </div>
                    <span className="text-gray-700">{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Detailed Feedback */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white p-6 rounded-2xl shadow-lg mb-8"
        >
          <h3 className="text-xl font-semibold text-gray-800 mb-4">
            Detailed Feedback
          </h3>
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-blue-50 border-l-4 border-l-blue-500">
              <h4 className="font-semibold text-gray-800 mb-2">
                Understanding
              </h4>
              <p className="text-gray-700">{result.understanding}</p>
            </div>
            <div className="p-4 rounded-xl bg-purple-50 border-l-4 border-l-purple-500">
              <h4 className="font-semibold text-gray-800 mb-2">Topic Grip</h4>
              <p className="text-gray-700">{result.topicGrip}</p>
            </div>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-center gap-4 mt-8">
          <button
            onClick={() => router.push("/dashboard")}
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
          >
            Back to Dashboard
          </button>
          {/* <button
            onClick={() => router.push("/practice")}
            className="bg-white text-indigo-600 border border-indigo-600 px-6 py-3 rounded-lg hover:bg-indigo-50 transition-colors font-medium"
          >
            Practice Again
          </button> */}
        </div>
      </div>
    </div>
  );
};

export default ResultPage;
