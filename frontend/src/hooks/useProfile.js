import { useQuery } from "react-query";
import { getProfile, getEmbedding, getCompletionHistory } from "../api/profile";

export const useProfile = () =>
  useQuery(["profile"], getProfile);

export const useEmbedding = () =>
  useQuery(["embedding"], getEmbedding);

// Profile.jsx needs both, but Dashboard.jsx only needs useProfile

export const useCompletionHistory = () =>
  useQuery(["completion-history"], getCompletionHistory);
