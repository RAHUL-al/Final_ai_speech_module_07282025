//@ts-check

"use client";

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useDispatch, useSelector } from "react-redux";
import { RootState, AppDispatch } from "@/store/index";
import { useRouter } from "next/navigation";
import {
  setClass,
  setSubject,
  setCurriculum,
  processFiles,
  resetUpload,
} from "@/store/slices/teacherSlice";
import Button from "@/components/UI/Button";
import Progress from "@/components/UI/Progress";
import {
  FileText,
  X,
  UploadCloud,
  ChevronDown,
  BookOpen,
  Bookmark,
  GraduationCap,
  LayoutGrid,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Teacher = () => {
  const dispatch = useDispatch<AppDispatch>();
  const router = useRouter();
  const {
    class: className,
    subject,
    curriculum,
    isLoading,
    error,
    progress,
    ocrResults,
  } = useSelector((state: RootState) => state.teacher);

  const [showResults, setShowResults] = useState(false);
  const [localFiles, setLocalFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setLocalFiles((prev: File[]) => [...prev, ...acceptedFiles]);
  }, []);

  const handleRemoveFile = (fileName: string) => {
    setLocalFiles((prev) => prev.filter((file) => file.name !== fileName));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".png", ".jpg"],
      "application/pdf": [".pdf"],
      "application/vnd.ms-powerpoint": [".ppt", ".pptx"],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handleSubmit = async () => {
    if (!className || !subject || !curriculum) {
      alert("Please select class, subject, and curriculum");
      return;
    }

    if (localFiles.length === 0) {
      alert("Please upload at least one file");
      return;
    }

    try {
      setIsProcessing(true);
      await dispatch(processFiles(localFiles));
    } catch (error) {
      console.error("Processing error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setLocalFiles([]);
    dispatch(resetUpload());
  };
  const handleChat = () => {
    router.push(`/chat`);
  };

  const classOptions = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"];
  const subjectOptions = ["Math", "Science", "History", "English", "Art"];
  const curriculumOptions = ["CBSE", "ICSE", "State Board", "IGCSE"];

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split(".").pop()?.toLowerCase();
    if (ext === "pdf") return <BookOpen className="text-red-500" />;
    if (["ppt", "pptx"].includes(ext || ""))
      return <LayoutGrid className="text-orange-500" />;
    if (["jpg", "jpeg", "png"].includes(ext || ""))
      return <Bookmark className="text-green-500" />;
    return <FileText className="text-blue-500" />;
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-10"
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-3">
          Teaching Material Upload
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload your teaching materials to extract text content using Google
          Vision OCR. Supported formats: PDF, PPT, JPG, PNG (max 10MB each)
        </p>
      </motion.div>

      {/* Context Selection */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100"
      >
        <div className="flex items-center mb-4">
          <GraduationCap className="h-6 w-6 text-indigo-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-800">
            Course Details
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700">
              Class
            </label>
            <div className="relative">
              <select
                value={className || ""}
                onChange={(e) => dispatch(setClass(e.target.value))}
                className="w-full text-black pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none"
              >
                <option value="">Select Class</option>
                {classOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span className="text-gray-400">📚</span>
              </div>
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <ChevronDown className="h-4 w-4 text-gray-400" />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700">
              Subject
            </label>
            <div className="relative">
              <select
                value={subject || ""}
                onChange={(e) => dispatch(setSubject(e.target.value))}
                className="w-full text-black pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none"
              >
                <option value="">Select Subject</option>
                {subjectOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span className="text-gray-400">🧪</span>
              </div>
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <ChevronDown className="h-4 w-4 text-gray-400" />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700">
              Curriculum
            </label>
            <div className="relative">
              <select
                value={curriculum || ""}
                onChange={(e) => dispatch(setCurriculum(e.target.value))}
                className="w-full text-black pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none"
              >
                <option value="">Select Curriculum</option>
                {curriculumOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span className="text-gray-400">🌐</span>
              </div>
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <ChevronDown className="h-4 w-4 text-gray-400" />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* File Dropzone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.5 }}
        className="mb-8"
      >
        <div
          {...getRootProps()}
          className={`rounded-2xl p-8 text-center cursor-pointer transition-all duration-300
            ${
              isDragActive
                ? "border-2 border-dashed border-blue-500 bg-blue-50 shadow-lg"
                : "border-2 border-dashed border-gray-300 hover:border-blue-400 bg-white hover:bg-blue-50 shadow-sm"
            }`}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={{ scale: isDragActive ? 1.05 : 1 }}
            className="flex flex-col items-center justify-center"
          >
            <div className="relative mb-4">
              <div
                className="absolute inset-0 bg-blue-100 rounded-full opacity-70 animate-ping"
                style={{ animationDuration: "1.5s" }}
              ></div>
              <UploadCloud className="mx-auto h-14 w-14 text-blue-500 relative z-10" />
            </div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive ? "Drop files here" : "Drag & drop files here"}
            </p>
            <p className="text-gray-500 mt-1">
              or <span className="text-blue-600 font-medium">browse files</span>
              from your device
            </p>
            <p className="text-sm text-gray-400 mt-3">
              Supports PDF, PPT, JPG, PNG (max 10MB each)
            </p>
          </motion.div>
        </div>
      </motion.div>

      {/* Uploaded Files Preview */}
      <AnimatePresence>
        {localFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-8"
          >
            <div className="flex items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">
                Selected Files
              </h2>
              <span className="ml-2 bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                {localFiles.length}
              </span>
            </div>
            <div className="space-y-3">
              {localFiles.map((file) => (
                <motion.div
                  key={file.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center justify-between p-4 bg-white rounded-lg border border-gray-100 shadow-sm hover:shadow transition-shadow"
                >
                  <div className="flex items-center">
                    <div className="p-2 bg-blue-50 rounded-lg mr-3">
                      {getFileIcon(file.name)}
                    </div>
                    <div>
                      <p className="text-black font-medium truncate max-w-xs">
                        {file.name}
                      </p>
                      <p className="text-gray-500 text-sm">
                        {(file.size / 1024 / 1024).toFixed(2)}MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile(file.name);
                    }}
                    className="p-2 rounded-full hover:bg-gray-100 text-gray-500 hover:text-red-500 transition-colors"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress and Results */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-8 p-6 bg-white rounded-xl shadow-sm border border-gray-100"
        >
          <div className="flex items-center mb-4">
            <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-blue-500 mr-3"></div>
            <h3 className="text-lg font-medium text-gray-800">
              Processing files with Google Vision OCR
            </h3>
          </div>
          <Progress
            value={progress}
            className="h-2.5 rounded-full bg-gray-200"
            indicatorClassName="bg-gradient-to-r from-blue-500 to-indigo-600"
          />
          <p className="mt-2 text-right text-sm text-gray-500">{progress}%</p>
        </motion.div>
      )}

      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-8 p-4 bg-red-50 border border-red-100 rounded-lg flex items-start"
        >
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-red-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">
              Error Processing Files
            </h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
          </div>
        </motion.div>
      )}

      <AnimatePresence>
        {Object.keys(ocrResults).length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-8 bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden"
          >
            <div className="flex justify-between items-center p-6 border-b border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800">
                OCR Results
              </h2>
              <Button
                variant="outline"
                onClick={() => setShowResults(!showResults)}
                className="flex items-center gap-2"
              >
                {showResults ? "Hide Results" : "Show Results"}
                <ChevronDown
                  className={`h-4 w-4 transition-transform ${
                    showResults ? "rotate-180" : ""
                  }`}
                />
              </Button>
            </div>

            {showResults && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-6 bg-gray-50"
              >
                <div className="space-y-5">
                  {Object.entries(ocrResults).map(([fileName, text]) => (
                    <div
                      key={fileName}
                      className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm"
                    >
                      <h3 className="font-medium mb-3 text-gray-700 flex items-center gap-2">
                        <FileText className="h-4 w-4 text-blue-500" />
                        {fileName}
                      </h3>
                      <div className="p-4 bg-gray-900 text-gray-100 rounded-md max-h-60 overflow-y-auto font-mono text-sm">
                        {text || "No text detected"}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex gap-4 justify-center pt-4">
        <Button
          onClick={handleSubmit}
          disabled={isProcessing || localFiles.length === 0}
          className="px-8 py-3 text-base font-medium bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 shadow-lg hover:shadow-blue-300 transition-all disabled:opacity-70"
        >
          {isProcessing ? (
            <span className="flex items-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Processing...
            </span>
          ) : (
            "Upload & Process"
          )}
        </Button>

        <Button
          variant="outline"
          onClick={handleReset}
          className="px-8 py-3 text-base text-black font-medium border-gray-300 hover:bg-gray-50"
        >
          Reset
        </Button>
        <Button
          variant="outline"
          onClick={handleChat}
          className="px-8 py-3 text-base font-medium hover:bg-blue-400 shadow-2xl border-blue-200 border-1 rounded-md"
        >
          Chat with us
        </Button>
      </div>
    </div>
  );
};

export default Teacher;
