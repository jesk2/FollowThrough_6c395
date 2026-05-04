import { useQuery, useMutation, useQueryClient } from "react-query";
import { getTasks, getPendingTasks, createTask, deleteTask } from "../api/tasks";

export const useTasks = () =>
  useQuery(["tasks"], getTasks);

export const usePendingTasks = () =>
  useQuery(["tasks", "pending"], getPendingTasks, {
    refetchInterval: 60_000,
  });

// usePendingTasks polls every 60 seconds with refetchInterval

export const useCreateTask = () => {
  const queryClient = useQueryClient();
  return useMutation(createTask, {
    onSuccess: () => {
      queryClient.invalidateQueries(["tasks"]);
    },
  });
};

export const useDeleteTask = () => {
  const queryClient = useQueryClient();
  return useMutation(deleteTask, {
    onSuccess: () => {
      queryClient.invalidateQueries(["tasks"]);
    },
  });
};
// the two mutations call invalidateQueries(["tasks"]) on success,
// so React Query should refetch both ["tasks"] and ["tasks", "pending"]
// automatically
