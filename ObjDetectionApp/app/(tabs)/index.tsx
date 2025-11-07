
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, Button, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';

const BACKEND_URL = "https://pubic-decadently-verda.ngrok-free.dev/detect";
const FRAME_INTERVAL = 800;

interface Obstacle {
  direction: string;
  distance: number;
  proximity: string;
  confidence: number;
  bbox: number[];
}

interface DetectionResult {
  detected: boolean;
  obstacles: Obstacle[];
  frame_id: number;
  total_detections: number;
  latency_ms: number;
  avg_latency_ms: number;
  status: string;
  realtime_mode: boolean;
  audio_message: string;
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [latestResult, setLatestResult] = useState<DetectionResult | null>(null);
  const cameraRef = useRef<CameraView>(null); 
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastAudioMessageRef = useRef<string>("");

 
  const speakMessage = async (message: string): Promise<void> => {
    try {
      if (message === lastAudioMessageRef.current) {
        return;
      }

      lastAudioMessageRef.current = message;

      let rate = 0.9;
      let pitch = 1.0;

      if (message.includes('CRITICAL')) {
        rate = 1.2;
        pitch = 1.2;
      } else if (message.includes('NEAR')) {
        rate = 1.0;
        pitch = 1.1;
      }

      await Speech.speak(message, {
        language: 'en',
        rate: rate,
        pitch: pitch,
        volume: 1.0,
      });

      console.log(`Spoke: ${message}`);
    } catch (error) {
      console.error('Speech error:', error);
    }
  };

  const captureAndSendFrame = async (): Promise<void> => {
    if (isProcessing || !cameraRef.current) return;

    try {
      setIsProcessing(true);
      
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
      });

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: photo.base64 }),
      });

      const result: DetectionResult = await response.json();

      if (response.ok) {
        setLatestResult(result);

        if (result.audio_message) {
          await speakMessage(result.audio_message);
        }

        console.log(' Detection:', {
          detected: result.detected,
          obstacles: result.total_detections,
          latency_ms: result.latency_ms,
          audio_message: result.audio_message,
        });
      } else {
        console.error('Backend error:', result);
      }
    } catch (error) {
      console.error(' Frame send error:', error);
      Alert.alert('Error', 'Failed to send frame to backend');
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (permission?.granted) {
      intervalRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL) as any;
      console.log('Started frame capture loop');
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current as any);
        console.log('Stopped frame capture loop');
      }
    };
  }, [permission?.granted]);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Camera permission required</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} />

      <View style={styles.overlay}>
        <View style={styles.statusBox}>
          <Text style={styles.statusText}>
            {isProcessing ? '⏳ Processing...' : 'Ready'}
          </Text>
          {latestResult && (
            <>
              <Text style={styles.resultText}>
                 Obstacles: {latestResult.total_detections}
              </Text>
              <Text style={styles.resultText}>
                  Latency: {latestResult.latency_ms}ms
              </Text>
              <Text style={styles.resultText}>
                 Status: {latestResult.status}
              </Text>
              {latestResult.obstacles && latestResult.obstacles.length > 0 && (
                <Text style={[
                  styles.resultText,
                  latestResult.obstacles[0].proximity === 'CRITICAL' && styles.critical,
                  latestResult.obstacles[0].proximity === 'NEAR' && styles.near,
                ]}>
                   {latestResult.obstacles[0].proximity} - {latestResult.obstacles[0].distance}m {latestResult.obstacles[0].direction}
                </Text>
              )}
            </>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  overlay: {
    position: 'absolute',
    top: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 15,
    borderRadius: 10,
  },
  statusBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#00ff00',
  },
  statusText: {
    color: '#00ff00',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resultText: {
    color: '#fff',
    fontSize: 13,
    marginBottom: 4,
  },
  critical: {
    color: '#ff0000',
    fontWeight: 'bold',
  },
  near: {
    color: '#ff9900',
    fontWeight: 'bold',
  },
  text: {
    color: '#fff',
  },
});
