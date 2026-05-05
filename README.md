# Ecomiles

A Flutter-based navigation app that computes and displays routes with minimum pollution exposure, instead of just shortest distance or time.

This project combines machine learning + geospatial routing to help users choose healthier travel paths within Gurugram, India.

## Overview

Traditional navigation systems (like Google Maps) optimize for:
1. shortest distance
2. fastest time

This system introduces a new optimization objective: **Minimize pollution exposure** along the route

The app predicts pollution levels across the road network and dynamically computes a least-pollution path.

## System Architecture
### Frontend (Flutter)
1. Map interface similar to Google Maps
2. Sends route requests to backend
3. Displays optimized route
4. Tracks user location in real time

### Backend API (Flask)
1. Accepts source & destination coordinates
2. date, time, day info
3. Returns list of route coordinates (lat, lon)

### Routing Engine
1. Built using **OSMNX** and **NetworkX**
2. Compute shortest path (baseline)
3. Predict pollution at nodes
4. Reweight edges using pollution
5. Compute least-pollution route

### Machine Learning Model
1. Model: XGBoost (via MultiOutputRegressor)
2. Predicts PM2.5 category
3. Features: Station ID, Time of day, Month, Date, Season, Weekday/Weekend
