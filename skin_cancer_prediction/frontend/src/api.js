import axios from 'axios';

// Create axios instance with base URL pointing to the backend API
const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
});

export default API;
