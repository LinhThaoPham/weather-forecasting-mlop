// ============================================================
// Atmospheric Weather Dashboard - Script
// Connects to: Data API (8001) + Forecast API (8000)
// ============================================================

const FORECAST_API = "http://localhost:8000";
const DATA_API = "http://localhost:8001";

let currentCity = "hanoi";
let currentMode = "hourly";
let forecastChart = null;

// City display names
const CITY_NAMES = {
  hanoi: "Hà Nội",
  danang: "Đà Nẵng",
  hcm: "TP. Hồ Chí Minh"
};

// OWM Weather Code → icon + description (Vietnamese)
// data_api converts OWM codes to WMO-like codes for compatibility
const WEATHER_CODES = {
  0: { icon: "light_mode", desc: "Trời quang" },
  1: { icon: "light_mode", desc: "Ít mây" },
  2: { icon: "cloud", desc: "Có mây" },
  3: { icon: "cloud", desc: "Nhiều mây" },
  45: { icon: "foggy", desc: "Sương mù" },
  48: { icon: "foggy", desc: "Sương mù đóng băng" },
  51: { icon: "rainy", desc: "Mưa phùn nhẹ" },
  53: { icon: "rainy", desc: "Mưa phùn" },
  55: { icon: "rainy", desc: "Mưa phùn dày" },
  61: { icon: "rainy", desc: "Mưa nhẹ" },
  63: { icon: "rainy", desc: "Mưa vừa" },
  65: { icon: "rainy", desc: "Mưa to" },
  71: { icon: "weather_snowy", desc: "Tuyết" },
  80: { icon: "rainy", desc: "Mưa rào" },
  81: { icon: "rainy", desc: "Mưa rào vừa" },
  82: { icon: "rainy", desc: "Mưa rào nặng" },
  95: { icon: "thunderstorm", desc: "Giông bão" },
  96: { icon: "thunderstorm", desc: "Giông bão + mưa đá" },
  99: { icon: "thunderstorm", desc: "Giông bão dữ dội" },
};

function getWeatherInfo(code) {
  if (WEATHER_CODES[code]) return WEATHER_CODES[code];
  if (code >= 95) return { icon: "thunderstorm", desc: "Giông bão" };
  if (code >= 71) return { icon: "weather_snowy", desc: "Tuyết" };
  if (code >= 51) return { icon: "rainy", desc: "Mưa" };
  if (code >= 2) return { icon: "cloud", desc: "Có mây" };
  return { icon: "light_mode", desc: "Trời quang" };
}


// ============== FETCH CURRENT WEATHER ==============
async function loadCurrentWeather(city) {
  try {
    const res = await fetch(`${DATA_API}/current?city=${city}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    const weatherInfo = getWeatherInfo(data.weather_code || 0);

    // Update hero section
    document.getElementById("heroCityName").textContent = `${data.city}, VN`;
    document.getElementById("heroTemp").textContent = `${Math.round(data.temperature)}°`;
    document.getElementById("heroWeatherDesc").textContent = weatherInfo.desc;
    document.getElementById("heroWeatherIcon").textContent = weatherInfo.icon;
    document.getElementById("heroHumidity").textContent = `💧 Độ ẩm: ${data.humidity}%`;
    document.getElementById("heroWind").textContent = `💨 Gió: ${data.wind_speed} km/h`;
    document.getElementById("heroCloud").textContent = `☁️ Mây: ${data.cloud_cover}%`;

    // Update conditions panel
    document.getElementById("condHumidity").textContent = `${data.humidity}%`;
    document.getElementById("condWind").textContent = `${data.wind_speed} km/h`;
    document.getElementById("condCloud").textContent = `${data.cloud_cover}%`;
    document.getElementById("condTemp").textContent = `${data.temperature}°C`;

    // Update header
    document.getElementById("headerCityName").textContent = data.city;
    document.getElementById("headerTimestamp").textContent = data.time;

    // API status
    document.getElementById("apiStatusBadge").textContent = "● Live";
    document.getElementById("apiStatusBadge").className = "text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-semibold";
    document.getElementById("modelStatus").textContent = "✓ Kết nối thành công";

    console.log(`✓ Current weather loaded for ${city}`);
  } catch (e) {
    console.error("Error loading current weather:", e);
    document.getElementById("apiStatusBadge").textContent = "● Offline";
    document.getElementById("apiStatusBadge").className = "text-xs bg-red-100 text-red-700 px-3 py-1 rounded-full font-semibold";
    document.getElementById("modelStatus").textContent = "✗ Không kết nối được";
    document.getElementById("heroWeatherDesc").textContent = "Không kết nối API";
  }
}


// ============== FETCH FORECAST ==============
async function loadForecast(city, mode) {
  try {
    const payload = {
      city: city,
      mode: mode,
      hours: mode === "hourly" ? 72 : 72,
      days: 3
    };

    const res = await fetch(`${FORECAST_API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();

    if (json.status !== "success") {
      throw new Error(json.message || "Prediction failed");
    }

    renderChart(json.data, mode);
    document.getElementById("predictionInfo").textContent =
      `${json.data.length} điểm dữ liệu • ${json.city_name || city}`;

    console.log(`✓ Forecast loaded: ${json.data.length} points (${mode})`);
  } catch (e) {
    console.error("Error loading forecast:", e);
    document.getElementById("predictionInfo").textContent = "Lỗi: " + e.message;
  }
}


// ============== RENDER CHART ==============
function renderChart(data, mode) {
  if (forecastChart) forecastChart.destroy();

  const ctx = document.getElementById("forecastChart").getContext("2d");

  let labels, values;

  if (mode === "hourly") {
    labels = data.map(item => {
      try {
        const d = new Date(item.ds);
        return d.toLocaleString("vi-VN", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
      } catch (e) { return item.ds; }
    });
    values = data.map(item => parseFloat(item.final_pred) || 0);
  } else {
    labels = data.map(item => {
      try {
        const d = new Date(item.ds);
        return d.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
      } catch (e) { return item.ds; }
    });
    values = data.map(item => parseFloat(item.final_pred) || 0);
  }

  const chartType = mode === "hourly" ? "line" : "bar";

  let gradient;
  if (chartType === "line") {
    gradient = ctx.createLinearGradient(0, 0, 0, 320);
    gradient.addColorStop(0, "rgba(0, 93, 167, 0.15)");
    gradient.addColorStop(1, "rgba(0, 93, 167, 0.0)");
  }

  forecastChart = new Chart(ctx, {
    type: chartType,
    data: {
      labels,
      datasets: [{
        label: mode === "hourly" ? "Nhiệt độ (°C)" : "Nhiệt độ TB (°C)",
        data: values,
        borderColor: "#005da7",
        backgroundColor: chartType === "line" ? gradient : "rgba(0, 118, 209, 0.7)",
        tension: 0.4,
        borderWidth: 2.5,
        fill: chartType === "line",
        pointBackgroundColor: "#005da7",
        pointBorderColor: "#ffffff",
        pointBorderWidth: 2,
        pointRadius: mode === "hourly" ? 0 : 5,
        pointHoverRadius: 6,
        borderRadius: chartType === "bar" ? 8 : 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index",
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(25, 28, 30, 0.9)",
          titleFont: { family: "Manrope", weight: "600" },
          bodyFont: { family: "Manrope" },
          cornerRadius: 12,
          padding: 12,
          callbacks: {
            label: (ctx) => `${ctx.parsed.y.toFixed(1)}°C`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          ticks: {
            color: "#707785",
            font: { family: "Manrope", size: 12 },
            callback: (v) => `${v}°`
          },
          grid: { color: "rgba(192, 199, 213, 0.2)" }
        },
        x: {
          ticks: {
            color: "#707785",
            font: { family: "Manrope", size: 11 },
            maxTicksLimit: mode === "hourly" ? 12 : 3,
            maxRotation: 45,
          },
          grid: { display: false }
        }
      }
    }
  });
}


// ============== UPDATE UI STATE ==============
function setActiveMode(mode) {
  currentMode = mode;
  const btnHourly = document.getElementById("btnHourly");
  const btnDaily = document.getElementById("btnDaily");

  if (mode === "hourly") {
    btnHourly.className = "px-4 py-1.5 rounded-full text-sm font-semibold bg-primary text-on-primary transition-all hover:opacity-90";
    btnDaily.className = "px-4 py-1.5 rounded-full text-sm font-semibold bg-surface-container-high text-on-surface-variant transition-all hover:bg-surface-dim";
    document.getElementById("chartTitle").textContent = "Dự báo theo giờ (72h)";
  } else {
    btnDaily.className = "px-4 py-1.5 rounded-full text-sm font-semibold bg-primary text-on-primary transition-all hover:opacity-90";
    btnHourly.className = "px-4 py-1.5 rounded-full text-sm font-semibold bg-surface-container-high text-on-surface-variant transition-all hover:bg-surface-dim";
    document.getElementById("chartTitle").textContent = "Nhiệt độ trung bình 3 ngày tới";
  }
}


// ============== EVENT LISTENERS ==============

document.getElementById("locationSelect")?.addEventListener("change", async (e) => {
  currentCity = e.target.value;
  await Promise.all([
    loadCurrentWeather(currentCity),
    loadForecast(currentCity, currentMode)
  ]);
});

document.getElementById("btnHourly")?.addEventListener("click", async () => {
  setActiveMode("hourly");
  await loadForecast(currentCity, "hourly");
});

document.getElementById("btnDaily")?.addEventListener("click", async () => {
  setActiveMode("daily");
  await loadForecast(currentCity, "daily");
});

document.getElementById("nav-hourly")?.addEventListener("click", async () => {
  setActiveMode("hourly");
  await loadForecast(currentCity, "hourly");
});

document.getElementById("nav-daily")?.addEventListener("click", async () => {
  setActiveMode("daily");
  await loadForecast(currentCity, "daily");
});


// ============== INIT ==============
window.addEventListener("DOMContentLoaded", async () => {
  const select = document.getElementById("locationSelect");
  currentCity = select?.value || "hanoi";

  console.log("🚀 Atmospheric Dashboard starting...");
  await Promise.all([
    loadCurrentWeather(currentCity),
    loadForecast(currentCity, currentMode)
  ]);
});