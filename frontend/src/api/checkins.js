import client from "./client";

export const submitCheckin = (payload) => client.post("/checkins", payload).then((r) => r.data);
export const getCheckinHistory = (page = 1) =>
  client.get(`/checkins/history?page=${page}`).then((r) => r.data);
