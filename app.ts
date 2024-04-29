import OpenAI from "openai";
import * as dotenv from "dotenv";
import fs from "fs";

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  try {
    const response = await openai.audio.transcriptions.create({
      model: "whisper-1",
      response_format: "srt",
      file: fs.createReadStream("src/audios/2023-12-18 01-33-22.mp3"),
    });

    console.log(response);
  } catch (e) {
    console.log(e);
  }
}

main();
