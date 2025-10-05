const API_BASE_URL = 'https://proje-1-kyt7.onrender.com/api/predict';
const GEOCODING_API_URL = 'https://geocoding-api.open-meteo.com/v1/search';
const API_HORIZON = 360;
const MAX_FORECAST_DAYS = 360;

let currentCity = null;
let currentForecastData = null;

document.addEventListener('DOMContentLoaded', () => {
  initializeDateInputs();
  loadFavorites();
  applyInitialTheme();
});

function isoLocalDate(d = new Date()) {
  return d.toLocaleDateString('en-CA');
}

function initializeDateInputs() {
  const todayStr = isoLocalDate(new Date());
  const max = new Date();
  max.setDate(max.getDate() + MAX_FORECAST_DAYS);
  const maxStr = isoLocalDate(max);

  const dateInput = document.getElementById('dateInput');
  dateInput.min = todayStr;
  dateInput.max = maxStr;
  dateInput.value = todayStr;
}

function applyInitialTheme() {
  const savedTheme = localStorage.getItem('theme');
  const isDark = savedTheme !== 'light';
  document.body.classList.toggle('light-mode', !isDark);

  const themeToggle = document.querySelector('.theme-toggle');
  if (themeToggle) themeToggle.textContent = isDark ? '☀ Light Mode' : '🌙 Dark Mode';

  const isPlanner = localStorage.getItem('plannerMode') === 'true';
  document.body.classList.toggle('planner-mode', isPlanner);
  const warningsDiv = document.getElementById('plannerWarnings');
  if (warningsDiv) warningsDiv.classList.toggle('hidden', !isPlanner);
  const plannerToggle = document.querySelector('.planner-toggle');
  if (plannerToggle) plannerToggle.textContent = isPlanner ? '✅ Planner ON' : '📋 Planner Mode';

  document.body.classList.toggle('image-background', localStorage.getItem('backgroundType') === 'image');
}

async function getCoordinates(city) {
  const geoUrl = `${GEOCODING_API_URL}?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;
  
  try {
    const response = await fetch(geoUrl);
    
    if (!response.ok) {
      throw new Error(`Geocoding HTTP Error Code: ${response.status}`);
    }

    const data = await response.json();

    if (data.results && data.results.length > 0) {
      const r = data.results[0];
      return { lat: parseFloat(r.latitude), lon: parseFloat(r.longitude) };
    }
    return null;
  } catch (error) {
    console.error('Geocoding API error:', error);
    const message = error.message.includes('400')
      ? 'Invalid city name or empty input.'
      : `API connection error: ${error.message}`;
      
    throw new Error(`Geocoding Error: ${message}`);
  }
}
async function searchWeather() {
  const city = document.getElementById('cityInput').value.trim();
  const selectedDateInput = document.getElementById('dateInput').value; 
  const selectedISO = selectedDateInput || isoLocalDate();

  if (!city) {
    alert('Please enter a valid city name.');
    return;
  }
  try {
    const location = await getCoordinates(city);
    if (!location) {
      alert(`Coordinates for "${city}" not found. Please try another city.`);

      return;
    }

    const todayISO = isoLocalDate(new Date());
    const leadDays = Math.ceil(
      (new Date(selectedISO + 'T00:00:00') - new Date(todayISO + 'T00:00:00')) / (24 * 3600 * 1000)
    );
    let horizon = API_HORIZON;
    if (leadDays >= 0) horizon = Math.max(API_HORIZON, leadDays + 1);

    const requestData = {
      lat: location.lat,
      lon: location.lon,
      target_date: selectedISO.replace(/-/g, ''), 
      horizon_days: horizon
    };

    const response = await fetch(API_BASE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error code: ${response.status}. Details: ${errorText || response.statusText}`);

    }

    const rawData = await response.json();
    const forecastList = rawData.gunluk || [];

    currentForecastData = { gunluk: forecastList };
    currentCity = city;

    const targetForecast = forecastList.find(f => {
      const apiDate = f.tarih || f.TARIH || f.date;
      return apiDate === selectedISO;
    });

    if (targetForecast) {
      updateUI(city, selectedISO, targetForecast);
    } else {
      const first = forecastList[0]?.tarih;
      const last  = forecastList.length ? forecastList[forecastList.length - 1].tarih : undefined; 
      document.getElementById('cityName').textContent = city;
      updateWeatherCard(
    { tmax: '...' },
    `Selected date is out of range. Valid range: ${first || '?'} – ${last || '?'}`
);
    }

  } catch (error) {
    console.error('Could not retrieve weather data:', error);
   alert(`An error occurred while fetching data: ${error.message}. Check the console.`);

  }
}

function updateUI(city, isoDate, forecast) {
  document.getElementById('cityName').textContent = city;
  document.getElementById('selectedDate').textContent = new Date(isoDate).toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  });
  updateWeatherCard(forecast);
  updatePlannerWarnings(forecast);
}

function updateWeatherCard(forecast, customDesc = null) {
  if (forecast.tmax === '...' || forecast.tmax === undefined) {
    document.getElementById('temperature').textContent = '...°C';
    document.getElementById('weatherDesc').textContent = customDesc || 'Loading...';
    document.getElementById('precipitation').textContent = '- %';
    document.getElementById('precipitationType').textContent = '-';
    document.getElementById('windSpeed').textContent = '- km/h';
    return;
  }

 const temp = (forecast.tmax !== 'Error') ? `${Math.round(forecast.tmax)}°C` : 'ERROR';
  document.getElementById('temperature').textContent = temp;

  const weatherDescElement = document.getElementById('weatherDesc');
  const oldBadge = weatherDescElement.querySelector('.data-source-badge');
  if (oldBadge) oldBadge.remove();

  let desc = customDesc || getWeatherDescription(forecast);
  weatherDescElement.textContent = desc;

  if (forecast.tmax_source && forecast.tmax_source.toUpperCase().includes('BLEND')) {
    const badge = document.createElement('span');
    badge.className = 'data-source-badge nasa-badge';
    badge.textContent = ' Climatology Blend';
    badge.title = 'Forecast is primarily based on climatology and long-term trend blend.';
    weatherDescElement.appendChild(badge);
  } else if (forecast.tmax_source && forecast.tmax_source.toUpperCase().includes('OPEN-METEO')) {
    const badge = document.createElement('span');
    badge.className = 'data-source-badge nwp-badge';
    badge.textContent = ' Meteorological Forecast';
    badge.title = 'Real-time forecast from numerical weather prediction models.';
    weatherDescElement.appendChild(badge);
  }

 document.getElementById('precipitation').textContent = `${forecast.yagis_iht ?? '-'}%`;
  document.getElementById('precipitationType').textContent = '-';
  document.getElementById('windSpeed').textContent = '- km/h';
}

function getWeatherDescription(forecast) {
  if (forecast.tmax === 'Error') return 'Data or API Connection Error.';
  let desc = `Max: ${Math.round(forecast.tmax)}°C. `;
  const p = Number(forecast.yagis_iht ?? 0);

  if (p >= 60) desc += `High chance of precipitation expected (${p}%). ☔`;
  else if (p >= 30) desc += `Some chance of precipitation possible (${p}%).`;
  else desc += 'Generally clear/sunny weather. ☀';

  return desc;
}

function toggleTheme() {
  const isLight = document.body.classList.toggle('light-mode');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
  const themeToggle = document.querySelector('.theme-toggle');
  if (themeToggle) themeToggle.textContent = isLight ? '🌙 Dark Mode' : '☀ Light Mode';
}

function togglePlanner() {
  const isPlanner = document.body.classList.toggle('planner-mode');
  const warnings = document.getElementById('plannerWarnings');
  warnings.classList.toggle('hidden', !isPlanner);
  localStorage.setItem('plannerMode', isPlanner);
  const plannerToggle = document.querySelector('.planner-toggle');
  if (plannerToggle) plannerToggle.textContent = isPlanner ? '✅ Planner ON' : '📋 Planner Mode';

  if (isPlanner && currentForecastData) {
    const selectedISO = document.getElementById('dateInput').value || isoLocalDate();
    const targetForecast = (currentForecastData.gunluk || []).find(
      f => (f.tarih || f.TARIH || f.date) === selectedISO
    );
    if (targetForecast) updatePlannerWarnings(targetForecast);
  }
}

function toggleBackgroundType() {
  const isImageBg = document.body.classList.toggle('image-background');
  localStorage.setItem('backgroundType', isImageBg ? 'image' : 'color');
  const backgroundToggle = document.querySelector('.background-toggle');
  if (backgroundToggle) backgroundToggle.textContent = isImageBg ? '🎨 Background Color' : '🖼 Background Images';
}

function loadFavorites() {
  const favorites = JSON.parse(localStorage.getItem('novaPulseFavorites')) || [];
  const container = document.getElementById('favoriteCities');
  container.innerHTML = '';

  if (favorites.length === 0) {
    container.innerHTML = '<p class="no-favorites">No favorites added yet.</p>';
    return;
  }

  favorites.forEach(city => {
    const cityDiv = document.createElement('div');
    cityDiv.className = 'favorite-item';
    cityDiv.innerHTML = `
      <span onclick="selectFavorite('${city}')">${city}</span>
      <button onclick="removeFavorite('${city}')" class="remove-btn">×</button>
    `;
    container.appendChild(cityDiv);
  });
}

function addToFavorites() {
  if (!currentCity || currentCity === 'Loading...' || currentCity === 'Error') {
    alert('Please search for a valid city first.');
    return;
  }

  let favorites = JSON.parse(localStorage.getItem('novaPulseFavorites')) || [];
  if (!favorites.includes(currentCity)) {
    favorites.push(currentCity);
    localStorage.setItem('novaPulseFavorites', JSON.stringify(favorites));
    loadFavorites();
    alert(`${currentCity} added to favorites!`);
  } else {
    alert(`${currentCity} is already in your favorites.`);
  }
}

function selectFavorite(city) {
  document.getElementById('cityInput').value = city;
  searchWeather();
}

function removeFavorite(cityToRemove) {
  let favorites = JSON.parse(localStorage.getItem('novaPulseFavorites')) || [];
  favorites = favorites.filter(city => city !== cityToRemove);
  localStorage.setItem('novaPulseFavorites', JSON.stringify(favorites));
  loadFavorites();
}

function updatePlannerWarnings(forecast) {
  const container = document.getElementById('warningsContainer');
  container.innerHTML = '';
  const warnings = [];

  if (Number(forecast.tmax) >= 25) {
    warnings.push({ text: "Sunscreen and light clothing recommended (High Temperature).", level: 'success' });
  }

  const p = Number(forecast.yagis_iht ?? 0);
  if (p >= 80) {
    warnings.push({ text: "High chance of heavy precipitation! Bring an umbrella. ☔", level: 'danger' });
  } else if (p < 30) {
    warnings.push({ text: "Perfect day for outdoor activities! No precipitation expected. ☀", level: 'success' });
  }

  if (warnings.length === 0) {
    container.innerHTML = '<p>No special planner warnings for today. Have a safe and enjoyable day!</p>';
  } else {
    warnings.forEach(w => {
      const el = document.createElement('p');
      el.className = `warning-level-${w.level}`;
      el.textContent = w.text;
      container.appendChild(el);
    });
  }
}
