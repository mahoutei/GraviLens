import { GoogleGenAI } from "@google/genai";
import { GRAVIBOT_SYSTEM_INSTRUCTION } from "../constants";

let client: GoogleGenAI | null = null;

const getClient = () => {
  if (!client) {
    const apiKey = process.env.API_KEY;
    if (apiKey) {
      client = new GoogleGenAI({ apiKey });
    } else {
      console.warn("API_KEY not found in environment variables.");
    }
  }
  return client;
};

export const sendMessageToGraviBot = async (history: { role: string; content: string }[], userMessage: string): Promise<string> => {
  const ai = getClient();
  if (!ai) return "Error: API Key configuration missing.";

  try {
    // Create a chat session
    const chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      config: {
        systemInstruction: GRAVIBOT_SYSTEM_INSTRUCTION,
        temperature: 0.7,
      },
      history: history.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.content }]
      }))
    });

    const result = await chat.sendMessage({ message: userMessage });
    return result.text || "I apologize, I couldn't generate a response.";
  } catch (error) {
    console.error("GraviBot Error:", error);
    return "Error connecting to the neural mainframe. Please try again later.";
  }
};