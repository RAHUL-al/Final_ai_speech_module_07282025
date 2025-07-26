import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { apiGet } from "@/components/services/api";


export const initializePractice = createAsyncThunk(
  "practice/initialize",
  async (paragraph: string, { rejectWithValue }) => {
    try {
      return { paragraph };
    } catch (error) {
      return rejectWithValue(
        error.response?.data?.message || "Failed to initialize practice"
      );
    }
  }
);

export const fetchOverallScoring = createAsyncThunk(
  "practice/fetchOverallScoring",
  async (essayId: string, { rejectWithValue }) => {
    try {
      const response = await apiGet(
        `/overall-scoring-by-id?essay_id=${essayId}`
      );
      return response;
    } catch (error) {
      return rejectWithValue(
        error.response?.data?.message || "Failed to fetch overall scoring"
      );
    }
  }
);

const initialState = {
  isRecording: false,
  isAnalyzing: false,
  analysisResults: null,
  error: null,
  paragraph: "",
  audioChunks: [],
  socket: null,
  mediaRecorder: null,
  stream: null,
};

const practiceSlice = createSlice({
  name: "practice",
  initialState,
  reducers: {
    startRecording: (state) => {
      state.isRecording = true;
    },
    stopRecording: (state) => {
      state.isRecording = false;
      if (state.mediaRecorder) {
        state.mediaRecorder.stop();
      }
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
      }
      if (state.socket) {
        state.socket.close();
      }
    },
    addAudioChunk: (state, action) => {
      state.audioChunks.push(action.payload);
    },
    setAnalysisResults: (state, action) => {
      state.analysisResults = action.payload;
      state.isAnalyzing = false;
    },
    setSocket: (state, action) => {
      state.socket = action.payload;
    },
    setMediaRecorder: (state, action) => {
      state.mediaRecorder = action.payload;
    },
    setStream: (state, action) => {
      state.stream = action.payload;
    },
    resetPracticeState: (state) => {
      Object.assign(state, initialState);
    },
     setAnalyzing: (state, action) => {
    state.isAnalyzing = action.payload;
  },

  },
  extraReducers: (builder) => {
    builder
      .addCase(initializePractice.pending, (state) => {
        state.isAnalyzing = true;
        state.error = null;
      })
      .addCase(initializePractice.fulfilled, (state, action) => {
        state.paragraph = action.payload.paragraph;
        state.isAnalyzing = false;
      })
      .addCase(initializePractice.rejected, (state, action) => {
        state.isAnalyzing = false;
        state.error = action.payload;
      })
      .addCase(fetchOverallScoring.pending, (state) => {
        state.isAnalyzing = true;
        state.error = null;
      })
      .addCase(fetchOverallScoring.fulfilled, (state, action) => {
        state.isAnalyzing = false;
        state.analysisResults = action.payload;
      })
      .addCase(fetchOverallScoring.rejected, (state, action) => {
        state.isAnalyzing = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  startRecording,
  stopRecording,
  addAudioChunk,
  setAnalysisResults,
  setSocket,
  setMediaRecorder,
  setStream,
  resetPracticeState,
} = practiceSlice.actions;

export default practiceSlice.reducer;
