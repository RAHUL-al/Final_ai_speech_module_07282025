// src/components/Chat/Chat.tsx

"use client";

import React, { useState, useEffect, useRef } from "react";
import { useAppDispatch, useAppSelector } from "@/store";
import { resetError, sendChatMessage } from "@/store/slices/chatSlice";
import { FiArrowUp, FiRefreshCw, FiX } from "react-icons/fi";
import { FaRobot, FaUser, FaBook, FaGraduationCap } from "react-icons/fa";
import Markdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";

const Chat = () => {
  const dispatch = useAppDispatch();
  const { status, error } = useAppSelector((state) => state.chat);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputValue, setInputValue] = useState("");
  const [curriculum, setCurriculum] = useState("");
  const [subjectInput, setSubjectInput] = useState("");
  const [messages, setMessages] = useState<
    Array<{
      id: string;
      text: string;
      sender: "user" | "ai";
      timestamp: Date;
    }>
  >([]);

  const [isCurriculumVisible, setIsCurriculumVisible] = useState(true);

  const isLoading = status === "loading";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!inputValue.trim() || !curriculum || !subjectInput || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputValue,
      sender: "user" as const,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue("");

    try {
      // Send to API
      const result = await dispatch(
        sendChatMessage({
          question: currentInput,
          curriculum: curriculum,
          subject: subjectInput,
        })
      ).unwrap();

      // Add AI response
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          text: result.answer,
          sender: "ai" as const,
          timestamp: new Date(),
        },
      ]);
      if (isCurriculumVisible) {
        setIsCurriculumVisible(false);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          text: "Sorry, I encountered an error. Please try again.",
          sender: "ai" as const,
          timestamp: new Date(),
        },
      ]);
      console.log("error", err);
    }
  };

  const messageVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  };

  const formVariants = {
    hidden: { height: 0, opacity: 0 },
    visible: { height: "auto", opacity: 1, transition: { duration: 0.3 } },
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl shadow-2xl overflow-hidden border border-gray-200">
      {/* Header with subtle gradient */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-5 text-white shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="bg-white/20 p-2 rounded-lg mr-3">
              <FaRobot className="text-xl" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Study Assistant</h1>
              <p className="text-indigo-200 text-sm">
                AI-powered learning companion
              </p>
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsCurriculumVisible(!isCurriculumVisible)}
            className="bg-white/10 hover:bg-white/20 rounded-full px-4 py-2 text-sm flex items-center transition-all"
          >
            <FaBook className="mr-2" />
            {isCurriculumVisible ? "Hide Details" : "Show Details"}
          </motion.button>
        </div>
      </div>

      {/* Curriculum and Subject Inputs */}
      <AnimatePresence>
        {isCurriculumVisible && (
          <motion.div
            variants={formVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            className="bg-white border-b border-gray-200"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                  <FaGraduationCap className="mr-2 text-indigo-600" />
                  Curriculum
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={curriculum}
                    onChange={(e) => setCurriculum(e.target.value)}
                    placeholder="e.g., ICSE, CBSE"
                    className="w-full text-black px-4 py-3 pl-10 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent shadow-sm"
                    required
                  />
                  <FaGraduationCap className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                  <FaBook className="mr-2 text-purple-600" />
                  Subject
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={subjectInput}
                    onChange={(e) => setSubjectInput(e.target.value)}
                    placeholder="e.g., Mathematics"
                    className="w-full text-black px-4 py-3 pl-10 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent shadow-sm"
                    required
                  />
                  <FaBook className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-5 bg-gradient-to-b from-white to-gray-50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl w-24 h-24 mb-6 flex items-center justify-center shadow-lg"
            >
              <FaRobot className="text-5xl text-indigo-600" />
            </motion.div>
            <motion.h3
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="text-2xl font-bold text-gray-800 mb-3"
            >
              Welcome to Study Assistant!
            </motion.h3>
            <motion.p
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="text-gray-600 max-w-md"
            >
              Enter your curriculum and subject, then ask your first question to
              get started with AI-powered learning help.
            </motion.p>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="mt-8 bg-indigo-50 rounded-xl p-4 max-w-md border border-indigo-100"
            >
              <h4 className="font-medium text-indigo-800 mb-2 flex items-center">
                <FiRefreshCw className="mr-2" />
                Try asking:
              </h4>
              <ul className="text-left text-gray-700 space-y-1 text-sm">
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">•</span>
                  {"Explain quadratic equations in simple terms"}
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">•</span>
                  {"What's the difference between mitosis and meiosis?"}
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">•</span>
                  {"Help me understand Newton's laws of motion"}
                </li>
              </ul>
            </motion.div>
          </div>
        ) : (
          <div className="space-y-6 pb-4">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  variants={messageVariants}
                  initial="hidden"
                  animate="visible"
                  exit={{ opacity: 0, x: message.sender === "user" ? 50 : -50 }}
                  className={`flex ${
                    message.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-3xl px-5 py-4 shadow-sm ${
                      message.sender === "user"
                        ? "bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-br-none"
                        : "bg-white border border-gray-200 rounded-bl-none"
                    }`}
                  >
                    <div className="flex items-center mb-2">
                      {message.sender === "ai" ? (
                        <div className="mr-2 bg-indigo-100 p-1 rounded-lg">
                          <FaRobot className="text-indigo-600" />
                        </div>
                      ) : (
                        <div className="mr-2 bg-indigo-500 p-1 rounded-lg">
                          <FaUser className="text-white" />
                        </div>
                      )}
                      <span className="text-xs font-semibold">
                        {message.sender === "ai" ? "Study Assistant" : "You"}
                      </span>
                      <span
                        className={`ml-auto text-xs ${
                          message.sender === "user"
                            ? "text-indigo-200"
                            : "text-gray-500"
                        }`}
                      >
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                    <div
                      className={`prose prose-sm max-w-none ${
                        message.sender === "user"
                          ? "text-indigo-50"
                          : "text-gray-700"
                      }`}
                    >
                      <Markdown>{message.text}</Markdown>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-white border border-gray-200 rounded-3xl rounded-bl-none px-5 py-4 max-w-[85%]">
                  <div className="flex items-center mb-2">
                    <div className="mr-2 bg-indigo-100 p-1 rounded-lg">
                      <FaRobot className="text-indigo-600" />
                    </div>
                    <span className="text-xs font-semibold">
                      Study Assistant
                    </span>
                  </div>
                  <div className="flex items-center">
                    <div className="flex space-x-1">
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          repeat: Infinity,
                          duration: 0.6,
                          delay: 0,
                        }}
                        className="w-2 h-2 bg-indigo-600 rounded-full"
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          repeat: Infinity,
                          duration: 0.6,
                          delay: 0.2,
                        }}
                        className="w-2 h-2 bg-indigo-600 rounded-full"
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          repeat: Infinity,
                          duration: 0.6,
                          delay: 0.4,
                        }}
                        className="w-2 h-2 bg-indigo-600 rounded-full"
                      />
                    </div>
                    <span className="ml-2 text-gray-600">Thinking...</span>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-gray-200 p-5 bg-white"
      >
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4 flex items-center"
          >
            <div className="text-red-700 flex-1">{error}</div>
            <button
              type="button"
              onClick={() => dispatch(resetError())}
              className="text-red-500 hover:text-red-700 ml-2"
            >
              <FiX size={18} />
            </button>
          </motion.div>
        )}

        <div className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask your question..."
              className="w-full text-black px-5 py-4 border border-gray-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none pr-12 shadow-sm"
              rows={1}
              style={{ minHeight: "56px", maxHeight: "150px" }}
              disabled={isLoading || !curriculum || !subjectInput}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = `${Math.min(target.scrollHeight, 150)}px`;
              }}
            />
            <div className="absolute bottom-3 right-3 text-xs text-gray-500 bg-white px-2 py-1 rounded">
              {inputValue.length}/500
            </div>
          </div>

          <motion.button
            type="submit"
            disabled={
              isLoading || !inputValue.trim() || !curriculum || !subjectInput
            }
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`flex-shrink-0 w-14 h-14 rounded-full flex items-center justify-center shadow-lg ${
              isLoading || !inputValue.trim() || !curriculum || !subjectInput
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
            }`}
          >
            {isLoading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <FiRefreshCw className="text-white text-xl" />
              </motion.div>
            ) : (
              <FiArrowUp className="text-white text-xl" />
            )}
          </motion.button>
        </div>

        <div className="mt-3 text-xs text-gray-500 flex justify-between">
          <span>
            {!curriculum || !subjectInput
              ? "Please enter curriculum and subject"
              : "Press Enter to send"}
          </span>
          <span className="text-gray-400">
            Study Assistant v1.0 • Powered by AI
          </span>
        </div>
      </form>
    </div>
  );
};

export default Chat;
// <div className="flex flex-col h-full max-w-4xl mx-auto bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl shadow-2xl overflow-hidden border border-gray-200">
//   {/* Header */}
//   <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-4 text-white">
//     <h1 className="text-2xl font-bold">Study Assistant</h1>
//     <p className="text-blue-200">Ask anything about your subjects</p>
//   </div>

//   <div className="flex p-4 space-x-3 border-b">
//     <div className="flex-1">
//       <label className="block text-sm font-medium text-gray-700 mb-1">
//         Curriculum
//       </label>
//       <input
//         type="text"
//         value={curriculum}
//         onChange={(e) => setCurriculum(e.target.value)}
//         placeholder="e.g., ICSE, CBSE"
//         className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
//         required
//       />
//     </div>
//     <div className="flex-1">
//       <label className="block text-sm font-medium text-gray-700 mb-1">
//         Subject
//       </label>
//       <input
//         type="text"
//         value={subjectInput}
//         onChange={(e) => setSubjectInput(e.target.value)}
//         placeholder="e.g., Mathematics"
//         className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
//         required
//       />
//     </div>
//   </div>

//   <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
//     {messages.length === 0 ? (
//       <div className="flex flex-col items-center justify-center h-full text-center p-8">
//         <div className="bg-gray-200 border-2 border-dashed rounded-xl w-16 h-16 mb-4 flex items-center justify-center">
//           <FaRobot className="text-3xl text-indigo-600" />
//         </div>
//         <h3 className="text-xl font-semibold text-gray-700 mb-2">
//           Welcome to Study Assistant!
//         </h3>
//         <p className="text-gray-500 max-w-md">
//           Enter your class and subject, then ask your first question to get
//           started with AI-powered learning help.
//         </p>
//       </div>
//     ) : (
//       <div className="space-y-4">
//         {messages.map((message) => (
//           <div
//             key={message.id}
//             className={`flex ${
//               message.sender === "user" ? "justify-end" : "justify-start"
//             }`}
//           >
//             <div
//               className={`max-w-[80%] rounded-2xl px-4 py-3 ${
//                 message.sender === "user"
//                   ? "bg-blue-600 text-white rounded-br-none"
//                   : "bg-white border border-gray-200 rounded-bl-none"
//               }`}
//             >
//               <div className="flex items-center mb-1">
//                 {message.sender === "ai" ? (
//                   <FaRobot className="mr-2 text-indigo-600" />
//                 ) : (
//                   <FaUser className="mr-2 text-blue-400" />
//                 )}
//                 <span className="text-xs font-medium">
//                   {message.sender === "ai" ? "Study Assistant" : "You"}
//                 </span>
//               </div>
//               <div className="prose prose-sm max-w-none">
//                 <Markdown>{message.text}</Markdown>
//               </div>
//               <div
//                 className={`text-xs mt-2 ${
//                   message.sender === "user"
//                     ? "text-blue-200"
//                     : "text-gray-500"
//                 }`}
//               >
//                 {message.timestamp.toLocaleTimeString([], {
//                   hour: "2-digit",
//                   minute: "2-digit",
//                 })}
//               </div>
//             </div>
//           </div>
//         ))}
//         {isLoading && (
//           <div className="flex justify-start">
//             <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-none px-4 py-3">
//               <div className="flex items-center">
//                 <FaRobot className="mr-2 text-indigo-600" />
//                 <span className="text-xs font-medium">Study Assistant</span>
//               </div>
//               <div className="flex items-center mt-2">
//                 <FiRefreshCw className="animate-spin text-indigo-600 mr-2" />
//                 <span>Thinking...</span>
//               </div>
//             </div>
//           </div>
//         )}
//         <div ref={messagesEndRef} />
//       </div>
//     )}
//   </div>

//   {/* Input Area */}
//   <form
//     onSubmit={handleSubmit}
//     className="border-t border-gray-200 p-4 bg-white"
//   >
//     {error && (
//       <div className="text-red-500 text-sm mb-2 flex items-center">
//         <span className="bg-red-100 px-2 py-1 rounded">{error}</span>
//         <button
//           type="button"
//           onClick={() => dispatch(resetError())}
//           className="ml-auto text-gray-500 hover:text-gray-700"
//         >
//           <FiX />
//         </button>
//       </div>
//     )}
//     <div className="flex">
//       <input
//         type="text"
//         value={inputValue}
//         onChange={(e) => setInputValue(e.target.value)}
//         placeholder="Ask your question..."
//         className="flex-1 px-4 py-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
//         disabled={isLoading || !curriculum || !subjectInput}
//       />
//       <button
//         type="submit"
//         disabled={
//           isLoading || !inputValue.trim() || !curriculum || !subjectInput
//         }
//         className={`px-5 rounded-r-lg flex items-center justify-center ${
//           isLoading || !inputValue.trim() || !curriculum || !subjectInput
//             ? "bg-gray-300 cursor-not-allowed"
//             : "bg-blue-600 hover:bg-blue-700"
//         }`}
//       >
//         {isLoading ? (
//           <FiRefreshCw className="animate-spin text-white" />
//         ) : (
//           <FiArrowUp className="text-white" />
//         )}
//       </button>
//     </div>
//     <div className="flex justify-between mt-2 text-xs text-gray-500">
//       <span>
//         {!curriculum || !subjectInput
//           ? "Please enter class and subject"
//           : "Press Enter to send"}
//       </span>
//       <span>{inputValue.length}/500</span>
//     </div>
//   </form>
// </div>
//   );
// };

// export default Chat;
