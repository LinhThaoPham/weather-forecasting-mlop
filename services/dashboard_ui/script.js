// ============================================================
// Atmospheric Weather Dashboard - Script
// Connects to: Data API (8001) + Forecast API (8000)
// ============================================================

// Auto-detect: Cloud Run URLs when deployed, localhost for local dev
const IS_CLOUD = window.location.hostname.includes("run.app");
const FORECAST_API = IS_CLOUD
  ? "https://forecast-api-217473815434.asia-southeast1.run.app"
  : "http://localhost:8000";
const DATA_API = IS_CLOUD
  ? "https://data-api-217473815434.asia-southeast1.run.app"
  : "http://localhost:8001";

let currentCity = "hanoi";
let currentMode = "hourly";
let forecastChart = null;

// City display names
const CITY_NAMES = {
  hanoi: "Hà Nội",
  hcm: "TP. Hồ Chí Minh",
  danang: "Đà Nẵng",
  haiphong: "Hải Phòng",
  nhatrang: "Nha Trang",
  dalat: "Đà Lạt"
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
      days: 7
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
            maxTicksLimit: mode === "hourly" ? 12 : 7,
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
    document.getElementById("chartTitle").textContent = "Nhiệt độ trung bình theo ngày";
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
  showWeatherDashboard();
  setActiveMode("hourly");
  await loadForecast(currentCity, "hourly");
});

document.getElementById("nav-daily")?.addEventListener("click", async () => {
  showWeatherDashboard();
  setActiveMode("daily");
  await loadForecast(currentCity, "daily");
});

document.getElementById("nav-dashboard")?.addEventListener("click", () => {
  showWeatherDashboard();
});

document.getElementById("nav-monitor")?.addEventListener("click", () => {
  showModelMonitor();
});


// ============== PAGE NAVIGATION ==============

function showWeatherDashboard() {
  document.querySelector("main")?.classList.remove("hidden");
  document.getElementById("modelMonitorSection")?.classList.add("hidden");
  setNavActive("nav-dashboard");
}

function showModelMonitor() {
  document.querySelector("main")?.classList.add("hidden");
  document.getElementById("modelMonitorSection")?.classList.remove("hidden");
  setNavActive("nav-monitor");
  loadMonitorData();
}

function setNavActive(activeId) {
  const navItems = ["nav-dashboard", "nav-hourly", "nav-daily", "nav-monitor"];
  navItems.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (id === activeId) {
      el.className = "w-full flex items-center space-x-3 text-blue-700 font-bold border-r-4 border-blue-600 py-3 px-3 hover:bg-blue-50/50 rounded-l-xl transition-all";
    } else {
      el.className = "w-full flex items-center space-x-3 text-slate-500 py-3 px-3 hover:bg-blue-50/50 hover:text-blue-800 rounded-xl transition-all";
    }
  });
}


// ============== MODEL MONITOR ==============

let lossChartHourly = null;
let lossChartDaily = null;
let cityMaeChart = null;
let registryData = null;

async function loadMonitorData() {
  try {
    const [regRes, histRes] = await Promise.all([
      fetch(`${DATA_API}/model/registry`),
      fetch(`${DATA_API}/model/history`)
    ]);

    registryData = await regRes.json();
    const historyData = await histRes.json();

    renderRegistryInfo(registryData);
    renderMetricsTable(registryData);
    renderCityMaeChart(registryData);

    if (historyData.entries && historyData.entries.length > 0) {
      populateVersionSelect(historyData.entries);
      renderLossCurves(historyData.entries[historyData.entries.length - 1]);
    } else {
      renderPlaceholderCurves();
    }

    console.log("✓ Monitor data loaded");
  } catch (e) {
    console.error("Error loading monitor data:", e);
  }
}

function renderRegistryInfo(reg) {
  if (!reg || !reg.models || reg.models.length === 0) return;

  const current = reg.models.find(m => m.version === reg.current_version) || reg.models[reg.models.length - 1];
  const metrics = current.metrics || {};

  document.getElementById("monitorVersion").textContent = current.version || "--";
  document.getElementById("monitorTrainedAt").textContent = current.trained_at
    ? new Date(current.trained_at).toLocaleString("vi-VN") : "--";

  const lstmH = metrics.lstm_hourly || {};
  const lstmD = metrics.lstm_daily || {};
  document.getElementById("monitorLstmHourlyMae").textContent = lstmH.mae != null ? `${lstmH.mae}°C` : "--";
  document.getElementById("monitorLstmDailyMae").textContent = lstmD.mae != null ? `${lstmD.mae}°C` : "--";

  const badge = document.getElementById("versionStatusBadge");
  if (current.decision === "accept") {
    badge.textContent = "✅ Active";
    badge.className = "text-xs px-3 py-1 rounded-full font-semibold bg-green-100 text-green-700";
  } else {
    badge.textContent = "❌ Rejected";
    badge.className = "text-xs px-3 py-1 rounded-full font-semibold bg-red-100 text-red-700";
  }
}

function populateVersionSelect(entries) {
  const select = document.getElementById("versionSelect");
  select.innerHTML = "";
  entries.slice().reverse().forEach((entry, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = entry.version + (i === 0 ? " (latest)" : "");
    select.appendChild(opt);
  });
  select.addEventListener("change", () => {
    const idx = parseInt(select.value);
    const reversed = entries.slice().reverse();
    if (reversed[idx]) renderLossCurves(reversed[idx]);
  });
}

function renderLossCurves(entry) {
  renderSingleLossCurve("lossChartHourly", entry.lstm_hourly, "LSTM Hourly", (c) => lossChartHourly = c);
  renderSingleLossCurve("lossChartDaily", entry.lstm_daily, "LSTM Daily", (c) => lossChartDaily = c);
}

function renderSingleLossCurve(canvasId, histData, label, setter) {
  // Destroy old chart
  if (canvasId === "lossChartHourly" && lossChartHourly) lossChartHourly.destroy();
  if (canvasId === "lossChartDaily" && lossChartDaily) lossChartDaily.destroy();

  const canvas = document.getElementById(canvasId);
  if (!canvas || !histData) return;
  const ctx = canvas.getContext("2d");

  const epochs = (histData.loss || []).map((_, i) => i + 1);

  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train Loss",
          data: histData.loss || [],
          borderColor: "#005da7",
          backgroundColor: "rgba(0, 93, 167, 0.08)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          label: "Val Loss",
          data: histData.val_loss || [],
          borderColor: "#ba5900",
          backgroundColor: "rgba(186, 89, 0, 0.08)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderDash: [6, 3],
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: {
          display: true,
          labels: { font: { family: "Manrope", size: 12 }, usePointStyle: true, pointStyle: "line" }
        },
        tooltip: {
          backgroundColor: "rgba(25, 28, 30, 0.9)",
          titleFont: { family: "Manrope", weight: "600" },
          bodyFont: { family: "Manrope" },
          cornerRadius: 10,
          padding: 10,
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Epoch", font: { family: "Manrope", size: 12 }, color: "#707785" },
          ticks: { color: "#707785", font: { family: "Manrope", size: 11 }, maxTicksLimit: 15 },
          grid: { display: false }
        },
        y: {
          title: { display: true, text: "Loss (MSE)", font: { family: "Manrope", size: 12 }, color: "#707785" },
          ticks: { color: "#707785", font: { family: "Manrope", size: 11 } },
          grid: { color: "rgba(192, 199, 213, 0.2)" }
        }
      }
    }
  });
  setter(chart);
}

function renderPlaceholderCurves() {
  ["lossChartHourly", "lossChartDaily"].forEach(id => {
    const ctx = document.getElementById(id)?.getContext("2d");
    if (!ctx) return;
    ctx.font = "14px Manrope";
    ctx.fillStyle = "#707785";
    ctx.textAlign = "center";
    ctx.fillText("Chưa có training history. Chạy retrain pipeline để tạo dữ liệu.", ctx.canvas.width / 2, 140);
  });
}

function renderMetricsTable(reg) {
  const tbody = document.getElementById("metricsTableBody");
  if (!tbody || !reg || !reg.models) return;
  tbody.innerHTML = "";

  reg.models.slice().reverse().forEach(model => {
    const m = model.metrics || {};
    const prophetH = m.prophet_hourly_hanoi || {};
    const lstmH = m.lstm_hourly || {};
    const lstmD = m.lstm_daily || {};

    const decisionIcon = model.decision === "accept"
      ? '<span class="text-green-600 font-semibold">✅ Accept</span>'
      : '<span class="text-red-600 font-semibold">❌ Reject</span>';

    const tr = document.createElement("tr");
    tr.className = "border-b border-outline-variant/20 hover:bg-surface-container-low/30 transition-colors";
    tr.innerHTML = `
      <td class="py-3 px-4 font-semibold text-primary">${model.version}</td>
      <td class="py-3 px-4 text-on-surface-variant">${model.trained_at ? model.trained_at.split("T")[0] : "--"}</td>
      <td class="py-3 px-4">${prophetH.mae != null ? prophetH.mae + "°C" : "--"}</td>
      <td class="py-3 px-4">${lstmH.mae != null ? lstmH.mae + "°C" : "--"}</td>
      <td class="py-3 px-4">${lstmD.mae != null ? lstmD.mae + "°C" : "--"}</td>
      <td class="py-3 px-4">${lstmH.val_loss != null ? lstmH.val_loss.toFixed(6) : "--"}</td>
      <td class="py-3 px-4">${decisionIcon}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCityMaeChart(reg) {
  if (cityMaeChart) cityMaeChart.destroy();

  if (!reg || !reg.models) return;
  const current = reg.models.find(m => m.version === reg.current_version) || reg.models[reg.models.length - 1];
  const m = current.metrics || {};

  const cities = ["hanoi", "hcm", "danang", "haiphong", "nhatrang", "dalat"];
  const cityLabels = ["Hà Nội", "HCM", "Đà Nẵng", "Hải Phòng", "Nha Trang", "Đà Lạt"];
  const hourlyMae = cities.map(c => (m[`prophet_hourly_${c}`] || {}).mae || 0);
  const dailyMae = cities.map(c => (m[`prophet_daily_${c}`] || {}).mae || 0);

  const ctx = document.getElementById("cityMaeChart")?.getContext("2d");
  if (!ctx) return;

  cityMaeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: cityLabels,
      datasets: [
        {
          label: "Hourly MAE (°C)",
          data: hourlyMae,
          backgroundColor: "rgba(0, 93, 167, 0.75)",
          borderRadius: 6,
          barPercentage: 0.4,
        },
        {
          label: "Daily MAE (°C)",
          data: dailyMae,
          backgroundColor: "rgba(186, 89, 0, 0.75)",
          borderRadius: 6,
          barPercentage: 0.4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { font: { family: "Manrope", size: 12 }, usePointStyle: true, pointStyle: "rectRounded" }
        },
        tooltip: {
          backgroundColor: "rgba(25, 28, 30, 0.9)",
          titleFont: { family: "Manrope", weight: "600" },
          bodyFont: { family: "Manrope" },
          cornerRadius: 10,
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}°C` }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "MAE (°C)", font: { family: "Manrope" }, color: "#707785" },
          ticks: { color: "#707785", font: { family: "Manrope", size: 11 } },
          grid: { color: "rgba(192, 199, 213, 0.2)" }
        },
        x: {
          ticks: { color: "#707785", font: { family: "Manrope", size: 12 } },
          grid: { display: false }
        }
      }
    }
  });
}


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