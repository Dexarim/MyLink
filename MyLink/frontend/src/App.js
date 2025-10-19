import { useRef, useState } from "react";
import { Card, Title, Text, Button, Group } from "@mantine/core";

export default function App() {
  const widgetRef = useRef(null);

  // --- Список вакансий ---
  const vacancies = [
    {
      id: 1,
      city: "Санкт-Петербург",
      experience: 2,
      post: "Дизайнер UX/UI",
      education: "Высшее художественное или ИТ",
      languages: ["русский", "английский"],
      salary: 600000,
      busyness: "полная",
      skills: ["Figma", "UX", "UI", "Прототипирование", "User Flow"],
      description:
        "Создание дизайн-макетов интерфейсов для веб и мобильных приложений. Совместная работа с продуктовой командой и фронтенд-разработчиками.",
    },
    {
      id: 2,
      city: "Москва",
      experience: 2,
      post: "Frontend-разработчик",
      education: "Бакалавр компьютерных наук",
      languages: ["русский", "английский"],
      salary: 700000,
      busyness: "полная",
      skills: ["React", "JavaScript", "TypeScript"],
      description: "Ищем фронтендера с опытом React и современных технологий.",
    },
    {
      id: 3,
      city: "Казань",
      experience: 3,
      post: "Системный администратор",
      education: "Среднее или высшее техническое",
      languages: ["русский"],
      salary: 500000,
      busyness: "полная",
      skills: ["Windows Server", "Linux", "Сети", "Active Directory"],
      description:
        "Поддержка серверной инфраструктуры, администрирование домена и рабочих станций, обеспечение безопасности сетей и резервного копирования.",
    },
  ];

  const [selectedVacancy, setSelectedVacancy] = useState(vacancies[0]);

  // --- Тестовое резюме ---
  const resume = {
    city: "Караганда",
    experience: 5,
    post: "junior python developer",
    education: "бакалавр компьютерных наук",
    languages: ["русский"],
    salary: 500000,
    busyness: "полная",
    skills: ["python", "django", "sql"],
    мотивация: "Хочу развиваться в Python backend",
  };

  // --- Отправка данных в SmartBotWidget ---
  const handleApply = (vacancy) => {
    setSelectedVacancy(vacancy);
    widgetRef.current?.setAttribute("data", JSON.stringify({ vacancy, resume }));
  };

  return (
    <div style={{ maxWidth: 650, margin: "40px auto", padding: 20 }}>
      <Title order={2} mb="md">
        Вакансии
      </Title>

      {/* Список карточек */}
      <Group direction="column" spacing="lg">
        {vacancies.map((vac) => (
          <Card
            key={vac.id}
            shadow="sm"
            padding="lg"
            radius="md"
            withBorder
            style={{
              border:
                selectedVacancy.id === vac.id
                  ? "2px solid #3b82f6"
                  : "1px solid #ddd",
              transition: "0.2s",
            }}
          >
            <Title order={4}>{vac.post}</Title>
            <Text mt="xs">Город: {vac.city}</Text>
            <Text size="sm" c="dimmed" mt="sm">
              {vac.description}
            </Text>
            <Text>Опыт работы: от {vac.experience} лет</Text>
            <Text mt="sm">
              💰 Зарплата: от {vac.salary.toLocaleString()} ₸
            </Text>
            <Button mt="md" fullWidth onClick={() => handleApply(vac)}>
              Откликнуться
            </Button>
          </Card>
        ))}
      </Group>

      {/* Сам виджет */}
      <smartbot-widget ref={widgetRef} api="http://127.0.0.1:8000"></smartbot-widget>
    </div>
  );
}
