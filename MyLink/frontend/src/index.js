import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider } from "@mantine/core";
import App from "./App";

// Подключаем глобальные стили
import "./index.css";
import "./App.css";

// Точка входа React-приложения
const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={{
        fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'",
        primaryColor: "blue",
        defaultRadius: "md",
        colors: {
          blue: [
            "#eff6ff",
            "#dbeafe",
            "#bfdbfe",
            "#93c5fd",
            "#60a5fa",
            "#3b82f6",
            "#2563eb",
            "#1d4ed8",
            "#1e40af",
            "#1e3a8a",
          ],
        },
      }}
    >
      <App />
    </MantineProvider>
  </React.StrictMode>
);
