import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import axios from "axios";

interface ScoringState {
  data: any;
  loading: boolean;
  error: string | null;
}

const initialState: ScoringState = {
  data: null,
  loading: false,
  error: null,
};

export const fetchOverallScoring = createAsyncThunk(
  "assistant/fetchOverallScoring",
  async (essay_id: string, { rejectWithValue }) => {
    try {
      const response = await axios.get(
        `https://llm.edusmartai.com/api/overall-scoring-by-id?essay_id=${essay_id}`
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

const assistantSlice = createSlice({
  name: "assistant",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchOverallScoring.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchOverallScoring.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchOverallScoring.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export default assistantSlice.reducer;
