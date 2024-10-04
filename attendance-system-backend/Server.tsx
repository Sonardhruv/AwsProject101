import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import * as faceapi from 'face-api.js';
import { Canvas, Image } from 'canvas';
import * as tf from '@tensorflow/tfjs-node';

dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

mongoose.connect(process.env.MONGODB_URI as string);

// Load face-api models
const loadModels = async () => {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');
};

loadModels();

// User model
const userSchema = new mongoose.Schema({
  name: String,
  email: String,
  department: String,
  faceDescriptor: Array,
});

const User = mongoose.model('User', userSchema);

// Attendance model
const attendanceSchema = new mongoose.Schema({
  userId: mongoose.Schema.Types.ObjectId,
  name: String,
  time: Date,
  status: String,
});

const Attendance = mongoose.model('Attendance', attendanceSchema);

// Register user
app.post('/api/users', async (req, res) => {
  try {
    const { name, email, department, image } = req.body;

    const img = await canvas.loadImage(image);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (!detection) {
      return res.status(400).json({ error: 'No face detected in the image' });
    }

    const faceDescriptor = detection.descriptor.toArray();

    const user = new User({ name, email, department, faceDescriptor });
    await user.save();

    res.status(201).json({ message: 'User registered successfully', userId: user._id });
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).json({ error: 'Error registering user' });
  }
});

// Mark attendance
app.post('/api/attendance', async (req, res) => {
  try {
    const { image } = req.body;

    const img = await canvas.loadImage(image);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (!detection) {
      return res.status(400).json({ error: 'No face detected in the image' });
    }

    const faceDescriptor = detection.descriptor;

    const users = await User.find();
    const labeledDescriptors = users.map(
      (user) => new faceapi.LabeledFaceDescriptors(user._id.toString(), [new Float32Array(user.faceDescriptor)])
    );

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    const match = faceMatcher.findBestMatch(faceDescriptor);

    if (match.distance < 0.6) {
      const user = await User.findById(match.label);
      if (user) {
        const attendance = new Attendance({
          userId: user._id,
          name: user.name,
          time: new Date(),
          status: 'Present',
        });
        await attendance.save();

        res.json({ message: 'Attendance marked successfully', user });
      } else {
        res.status(404).json({ error: 'User not found' });
      }
    } else {
      res.status(404).json({ error: 'Face not recognized' });
    }
  } catch (error) {
    console.error('Error marking attendance:', error);
    res.status(500).json({ error: 'Error marking attendance' });
  }
});

// Get attendance records
app.get('/api/attendance', async (req, res) => {
  try {
    const attendanceRecords = await Attendance.find().sort({ time: -1 });
    res.json(attendanceRecords);
  } catch (error) {
    console.error('Error fetching attendance records:', error);
    res.status(500).json({ error: 'Error fetching attendance records' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});