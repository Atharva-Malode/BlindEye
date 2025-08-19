import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";

export default function CamPage() {
  const [running, setRunning] = useState(false);
  const [socket, setSocket] = useState(null);
  const imgRef = useRef();
  const navigate = useNavigate();

  const speak = (text) => {
    console.log("🔊 Speaking:", text);
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.cancel(); // stop any ongoing speech
    speechSynthesis.speak(utterance);
  };

  const startFeed = () => {
    if (running) return;

    const ws = new WebSocket("ws://localhost:8000/ws/cam");

    ws.onopen = () => {
      console.log("✅ WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("📩 Received data:", data);

        // Update image
        if (imgRef.current && data.frame) {
          imgRef.current.src = "data:image/jpeg;base64," + data.frame;
        }

        // Speak detection if available
        if (data.detection) {
          console.log("🎯 Detection:", data.detection);
          speak(data.detection);
        }
      } catch (err) {
        console.error("❌ Error parsing message:", err, event.data);
      }
    };

    ws.onclose = () => {
      console.log("❌ WebSocket closed");
      setRunning(false);
    };

    ws.onerror = (err) => {
      console.error("⚠️ WebSocket error:", err);
    };

    setSocket(ws);
    setRunning(true);
  };

  const stopFeed = () => {
    if (socket) {
      socket.close();
      setSocket(null);
    }
    setRunning(false);
    if (imgRef.current) {
      imgRef.current.src = ""; // clear the image
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
      <div className="bg-white text-gray-800 rounded-2xl shadow-2xl p-8 flex flex-col items-center space-y-6">
        <h1 className="text-2xl font-bold">Camera Feed with AI Guidance</h1>
        <img
          ref={imgRef}
          alt="Webcam Feed"
          className="w-[640px] h-[480px] bg-black rounded-xl shadow-md"
        />
        <div className="flex space-x-4">
          <button
            onClick={startFeed}
            className="px-4 py-2 bg-green-500 text-white rounded-xl shadow hover:bg-green-600"
          >
            Start
          </button>
          <button
            onClick={stopFeed}
            className="px-4 py-2 bg-red-500 text-white rounded-xl shadow hover:bg-red-600"
          >
            Stop
          </button>
          <button
            onClick={() => navigate("/")}
            className="px-4 py-2 bg-indigo-500 text-white rounded-xl shadow hover:bg-indigo-600"
          >
            Back
          </button>
        </div>
      </div>
    </div>
  );
}
