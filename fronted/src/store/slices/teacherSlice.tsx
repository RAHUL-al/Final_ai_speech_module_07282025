// @ts-nocheck
import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { apiPost } from "@/components/services/api";

interface UploadState {
  class: string | null;
  subject: string | null;
  curriculum: string | null;
  isLoading: boolean;
  error: string | null;
  ocrResults: { [fileName: string]: string };
  progress: number;
}

const initialState: UploadState = {
  class: null,
  subject: null,
  curriculum: null,
  isLoading: false,
  error: null,
  ocrResults: {},
  progress: 0,
};

export const processFiles = createAsyncThunk(
  "teacher/processFiles",
  async (files: File[], { getState, rejectWithValue, dispatch }) => {
    // Accept files as argument
    try {
      const state = getState() as { teacher: UploadState };
      const { class: className, subject, curriculum } = state.teacher;

      const results: { [fileName: string]: string } = {};

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();

        formData.append("file", file);
        formData.append("student_class", className || "");
        formData.append("subject", subject || "");
        formData.append("curriculum", curriculum || "");

        const progress = Math.floor(((i + 1) / files.length) * 100);
        dispatch(setProgress(progress));

        const response = await apiPost("/upload/", formData);

        results[file.name] = response.data?.extracted_text || "No text";
      }

      return { results };
    } catch (error: unknown) {
      console.error("Upload error:", error);
      let errorMessage = "Processing failed";
      if (error.response) {
        errorMessage =
          error.response.data?.detail ||
          error.response.statusText ||
          `Server error: ${error.response.status}`;
      } else if (error.message) {
        errorMessage = error.message;
      }

      return rejectWithValue(errorMessage);
    }
  }
);

const teacherSlice = createSlice({
  name: "teacher",
  initialState,
  reducers: {
    setClass: (state, action: PayloadAction<string>) => {
      state.class = action.payload;
    },
    setSubject: (state, action: PayloadAction<string>) => {
      state.subject = action.payload;
    },
    setCurriculum: (state, action: PayloadAction<string>) => {
      state.curriculum = action.payload;
    },
    resetUpload: () => initialState,
    setProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(processFiles.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.progress = 0;
      })
      .addCase(processFiles.fulfilled, (state, action) => {
        state.isLoading = false;
        state.ocrResults = action.payload.results;
        state.progress = 100;
      })
      .addCase(processFiles.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.progress = 0;
      });
  },
});

export const { setClass, setSubject, setCurriculum, resetUpload, setProgress } =
  teacherSlice.actions;

export default teacherSlice.reducer;
// function dispatch(arg0: { payload: number; type: "teacher/setProgress" }) {
//   throw new Error("Function not implemented.");
// }
