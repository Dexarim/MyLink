import { useRef, useState, useEffect } from "react";
import {
  Title,
  Button,
  Loader,
  Card,
  Text,
  Group,
  Avatar,
  ScrollArea,
} from "@mantine/core";

const API_BASE = "http://127.0.0.1:8002";

export default function App() {
  const widgetRef = useRef(null);
  const [vacancies, setVacancies] = useState([]);
  const [resumes, setResumes] = useState([]);
  const [selectedResume, setSelectedResume] = useState(null);
  const [loading, setLoading] = useState(false);

  // --- Загружаем данные из API ---
  useEffect(() => {
    fetch(`${API_BASE}/vacancies/`)
      .then((r) => r.json())
      .then(setVacancies)
      .catch((e) => console.error("Ошибка при загрузке вакансий:", e));

    fetch(`${API_BASE}/resumes/`)
      .then((r) => r.json())
      .then(setResumes)
      .catch((e) => console.error("Ошибка при загрузке резюме:", e));
  }, []);

  // --- Создание заявки ---
  const handleApply = async (vacancy) => {
    if (!selectedResume) {
      alert("Выберите резюме перед откликом!");
      return;
    }

    setLoading(true);
    try {
      const resApp = await fetch(`${API_BASE}/applications/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_id: selectedResume.id,
          vacancy_id: vacancy.id,
        }),
      });

      const appData = await resApp.json();
      console.log("✅ Заявка создана:", appData);

      widgetRef.current?.setAttribute(
        "data",
        JSON.stringify({ vacancy, resume: selectedResume })
      );
    } catch (err) {
      console.error("Ошибка при создании заявки:", err);
      alert("Ошибка соединения с API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "60px auto",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "24px",
      }}
    >
      {/* ---------- Шапка ---------- */}
      <Group position="apart" align="center">
        <Title order={3}>DEMO HR Bot</Title>
        <Text size="sm" c="dimmed">
          {selectedResume
            ? `Активное резюме: ${selectedResume.name || "Без имени"}`
            : "Резюме не выбрано"}
        </Text>
      </Group>

      {/* ---------- Список резюме (горизонтальная полоса) ---------- */}
      <ScrollArea
        type="never"
        style={{
          whiteSpace: "nowrap",
          paddingBottom: "8px",
        }}
      >
        <Group gap="md" wrap="nowrap">
          {resumes.map((r) => (
            <Card
              key={r.id}
              p="sm"
              radius="md"
              withBorder
              shadow="sm"
              style={{
                cursor: "pointer",
                minWidth: "180px",
                backgroundColor:
                  selectedResume?.id === r.id ? "#e0f2fe" : "white",
                transition: "0.2s",
              }}
              onClick={() => setSelectedResume(r)}
            >
              <Group>
                <Avatar color="blue" radius="xl">
                  {r.name ? r.name[0].toUpperCase() : "?"}
                </Avatar>
                <div>
                  <Text fw={500} size="sm">
                    {r.name || "Без имени"}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {r.post}
                  </Text>
                </div>
              </Group>
            </Card>
          ))}
        </Group>
      </ScrollArea>

      {/* ---------- Список вакансий (в карточках) ---------- */}
      {vacancies.length === 0 ? (
        <Text c="dimmed">Загрузка вакансий...</Text>
      ) : (
        vacancies.map((vac) => (
          <Card
            key={vac.id}
            shadow="sm"
            padding="lg"
            radius="md"
            withBorder
            style={{
              transition: "0.2s",
              border:
                selectedResume && selectedResume.id
                  ? "1px solid #e2e8f0"
                  : "1px dashed #cbd5e1",
            }}
          >
            <Title order={4}>{vac.post}</Title>
            <Text mt="xs" size="sm" c="dimmed">
              {vac.city} — {vac.education}
            </Text>
            <Text size="sm" mt="sm">
              {vac.description}
            </Text>
            <Text mt="sm">💰 Зарплата: {vac.salary.toLocaleString()} ₸</Text>
            <Text size="sm" mt="sm">
              Навыки: {vac.skills.join(", ")}
            </Text>

            <Button
              mt="md"
              fullWidth
              onClick={() => handleApply(vac)}
              disabled={loading}
            >
              {loading ? <Loader color="white" size="sm" /> : "Откликнуться"}
            </Button>
          </Card>
        ))
      )}

      {/* ---------- SmartBot Widget ---------- */}
      <smartbot-widget ref={widgetRef} api="http://127.0.0.1:8000"></smartbot-widget>
    </div>
  );
}
