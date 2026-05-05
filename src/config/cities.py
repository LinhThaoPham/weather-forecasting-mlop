
CITIES = {
    "hanoi": {
        "lat": 21.0285,
        "lon": 105.8542,
        "name": "Hà Nội",
        "country": "VN"
    },
    "hcm": {
        "lat": 10.8231,
        "lon": 106.6297,
        "name": "TP. Hồ Chí Minh",
        "country": "VN"
    },
    "danang": {
        "lat": 16.0544,
        "lon": 108.2022,
        "name": "Đà Nẵng",
        "country": "VN"
    },
}
def get_city_coords(city: str = "hanoi"):
    """Get city info dict. Returns Hà Nội as default."""
    return CITIES.get(city.lower(), CITIES["hanoi"])
