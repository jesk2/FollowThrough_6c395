import { useQuery, useMutation, useQueryClient } from "react-query";
import { getCheckinHistory, submitCheckin } from "../api/checkins";

export const useCheckinHistory = (page = 1) =>
  useQuery(["checkins", "history", page], () => getCheckinHistory(page), {
    keepPreviousData: true,
    // when paging through check-in history, the old data stays visible while the new page loads instead of flashing a loading spinner
  });

export const useSubmitCheckin = () => {
  const queryClient = useQueryClient();
  return useMutation(submitCheckin, {
    onSuccess: () => {
      queryClient.invalidateQueries(["tasks", "pending"]);
      queryClient.invalidateQueries(["profile"]); // beta update
    },
  });
};
