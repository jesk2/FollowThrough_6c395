import client from "./client";

export const getProfile = () => client.get("/profile").then((r) => r.data);
export const getEmbedding = () => client.get("/profile/embedding").then((r) => r.data);
