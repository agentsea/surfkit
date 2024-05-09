// App.js
import React from "react";
import { BrowserRouter as Router } from "react-router-dom";
import AppRoutes from "./server/Routes";

function App() {
  return (
    <div className="bg-gray-100 min-h-screen">
      <Router>
        <AppRoutes />
      </Router>
    </div>
  );
}

export default App;
