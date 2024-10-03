import React from "react";
import { Route, Routes } from "react-router-dom";

import DesktopPage from "../pages/DesktopPage";
import ContainerDesktopPage from "../pages/ContainerDesktopPage";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<DesktopPage />} />
      <Route path="/container" element={<ContainerDesktopPage />} />
    </Routes>
  );
}
