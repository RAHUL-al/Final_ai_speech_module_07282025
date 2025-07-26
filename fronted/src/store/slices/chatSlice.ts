// src/lib/features/chat/chatSlice.ts
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

interface UserQuestion {
  question: string;
  curriculum: string;
  subject: string;
}

interface ChatError {
  message: string;
}

export const sendChatMessage = createAsyncThunk(
  "chat/sendMessage",
  async (userQuestion: UserQuestion, { rejectWithValue }) => {
    try {
      // const state = getState() as { teacher: UploadState };
      // const { class: studentClass, subject, curriculum } = state.teacher;

      const payload = {
        question: userQuestion.question,
        subject: userQuestion.subject || "",
        curriculum: userQuestion.curriculum || "",
      };

      const backendUrl =
        process.env.NEXT_PUBLIC_API_URL || "https://llm.edusmartai.com/api";

      const response = await fetch(`${backendUrl}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(`Invalid response: ${text.substring(0, 100)}`);
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Request failed");
      }

      return await response.json();
    } catch (error: unknown) {
      return rejectWithValue(
        (error as ChatError).message || "An error occurred"
      );
    }
  }
);
interface ChatState {
  status: "idle" | "loading" | "succeeded" | "failed";
  error: string | null;
}

const initialState: ChatState = {
  status: "idle",
  error: null,
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    resetError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(sendChatMessage.pending, (state) => {
        state.status = "loading";
        state.error = null;
      })
      .addCase(sendChatMessage.fulfilled, (state) => {
        state.status = "succeeded";
      })
      .addCase(sendChatMessage.rejected, (state, action) => {
        state.status = "failed";
        state.error = action.payload as string;
      });
  },
});

export const { resetError } = chatSlice.actions;
export default chatSlice.reducer;
