import client from "./client";

export const getTasks = () => client.get("/tasks").then((r) => r.data);
export const getPendingTasks = () => client.get("/tasks/pending").then((r) => r.data);
export const createTask = (payload) => client.post("/tasks", payload).then((r) => r.data);
export const deleteTask = (taskId) => client.delete(`/tasks/${taskId}`);
