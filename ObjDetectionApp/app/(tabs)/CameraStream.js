import React, { useEffect, useRef, useState } from "react";
import { View, Button, Text } from "react-native";
import { Camera, CameraType } from "expo-camera";

export default function CameraStream() {
    const [permission, requestPermission] = Camera.useCameraPermissions();
    const cameraRef = useRef(null);
    const [streaming, setStreaming] = useState(false);
    const intervalRef = useRef(null);

    const BACKEND_URL = "https://pubic-decadently-verda.ngrok-free.dev/detect";

    useEffect(() => {
        if (!permission) requestPermission();
    }, []);

    const sendFrame = async () => {
        if (!cameraRef.current) return;
        const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.4,
            skipProcessing: true,
        });

        try {
            const res = await fetch(BACKEND_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: photo.base64 }),
            });
            const data = await res.json();
            console.log("Detection:", data);
        } catch (err) {
            console.error("Error:", err);
        }
    };

    const startStreaming = () => {
        if (streaming) return;
        setStreaming(true);
        intervalRef.current = setInterval(sendFrame, 1000); // every 1s
    };

    const stopStreaming = () => {
        setStreaming(false);
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    };

    return (
        <View style={{ flex: 1 }}>
            <Camera ref={cameraRef} style={{ flex: 1 }} type={CameraType.back} />
            <Button
                title={streaming ? "Stop Streaming" : "Start Streaming"}
                onPress={streaming ? stopStreaming : startStreaming}
            />
        </View>
    );
}
