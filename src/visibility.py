import marimo

__generated_with = "0.11.2"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime
    import httpx

    import polars as pl
    import marimo as mo
    import numpy as np

    from vega_datasets import data as vega_data

    from sgp4.api import Satrec, jday
    from sgp4.api import WGS72

    from skyfield.api import EarthSatellite, load, Topos
    from skyfield.api import load, wgs84, N, S, E, W

    ts = load.timescale()
    return (
        E,
        EarthSatellite,
        N,
        S,
        Satrec,
        Topos,
        W,
        WGS72,
        datetime,
        httpx,
        jday,
        load,
        mo,
        np,
        pl,
        ts,
        vega_data,
        wgs84,
    )


@app.cell
def _(mo):
    mo_btn_refresh = mo.ui.run_button(
        label="Refresh Data",
        tooltip="Fetch the latest data.",
    )
    mo.hstack([mo_btn_refresh], justify="end")
    return (mo_btn_refresh,)


@app.cell
def _(Satrec, httpx, pl):
    # Fetch the TLE text from CelesTrak
    _url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    _response = httpx.get(_url)
    _response.raise_for_status()
    _tle_text = _response.text

    # Split the TLE text into lines and remove any blank lines
    _lines = [line.strip() for line in _tle_text.splitlines() if line.strip()]

    # Verify that we have groups of 3 lines (satellite name + two TLE lines)
    if len(_lines) % 3 != 0:
        raise ValueError(
            "Unexpected TLE format: the number of lines is not a multiple of 3."
        )

    _tle_entries = []
    _satellites = []

    # Process each group of 3 lines
    for _i in range(0, len(_lines), 3):
        _name = _lines[_i]
        _tle_line1 = _lines[_i + 1]
        _tle_line2 = _lines[_i + 2]

        # Extract the NORAD catalog number from TLE line 1 (characters 3 to 7)
        _norad_cat = int(_tle_line1[2:7])

        # Append the TLE entry to our list
        _tle_entries.append(
            {
                "OBJECT_NAME": _name,
                "TLE_LINE1": _tle_line1,
                "TLE_LINE2": _tle_line2,
                "NORAD_CAT_ID": _norad_cat,
            }
        )

        # Create a satellite object from the TLE lines and add it to the list
        _sat = Satrec.twoline2rv(_tle_line1, _tle_line2)
        _satellites.append(_sat)

    # Create a Polars DataFrame from the TLE entries; this is exported.
    df_tles = pl.DataFrame(_tle_entries)

    # Optionally, you can also export _satellites if needed.
    df_tles
    return (df_tles,)


@app.cell
def _(datetime, mo):
    start_date = mo.ui.date(
        value=datetime.datetime.now().date(), label="Start Date"
    )

    end_date = mo.ui.date(value=datetime.datetime.now().date(), label="End Date")

    # Kihei, Maui approximate coordinates:
    sensor_lat = mo.ui.number(value=20.7649, label="Sensor Latitude (°)")
    sensor_lon = mo.ui.number(value=-156.3311, label="Sensor Longitude (°)")
    sensor_alt = mo.ui.number(
        value=0.01,  # approximately 10 m above sea level
        label="Sensor Altitude (km)",
    )

    sensor_name = mo.ui.text(
        value="My Sensor", placeholder="Enter sensor name", label="Sensor Name"
    )

    user_date = mo.hstack([start_date, end_date], justify="start")
    user_site = mo.hstack([sensor_lat, sensor_lon, sensor_alt], justify="start")
    user_input = mo.vstack([user_date, user_site])

    user_input
    return (
        end_date,
        sensor_alt,
        sensor_lat,
        sensor_lon,
        sensor_name,
        start_date,
        user_date,
        user_input,
        user_site,
    )


@app.cell
def _(
    E,
    EarthSatellite,
    N,
    S,
    W,
    datetime,
    df_tles,
    end_date,
    load,
    np,
    sensor_alt,
    sensor_lat,
    sensor_lon,
    start_date,
    ts,
    wgs84,
):
    # Load Skyfield timescale
    _planets = load("de421.bsp")
    _earth = _planets["earth"]

    # Build a list of satellite objects from the TLE DataFrame
    satellites_sky = []
    for _row in df_tles.to_dicts():
        satellites_sky.append(
            EarthSatellite(
                _row["TLE_LINE1"], _row["TLE_LINE2"], _row["OBJECT_NAME"], ts
            )
        )

    # Define the start and end times using the user-provided dates (start_date and end_date are date objects)
    _t0 = ts.utc(
        start_date.value.year,
        start_date.value.month,
        start_date.value.day,
        0,
        0,
        0,
    )
    _t1 = ts.utc(
        end_date.value.year, end_date.value.month, end_date.value.day, 23, 59, 59
    )

    # Create an observer location using wgs84.latlon and the sensor inputs.
    # Use N if latitude is positive, S if negative; W if longitude is negative, E if positive.
    observer = wgs84.latlon(
        sensor_lat.value * (N if sensor_lat.value >= 0 else S),
        abs(sensor_lon.value) * (W if sensor_lon.value < 0 else E),
        elevation_m=sensor_alt.value
        * 1000,  # sensor_alt is in km, convert to meters
    )

    # Generate time samples at 1-minute intervals between _t0 and _t1.
    _t0_dt = _t0.utc_datetime()
    _t1_dt = _t1.utc_datetime()
    _total_minutes = int((_t1_dt - _t0_dt).total_seconds() // 60)
    _sample_datetimes = [
        _t0_dt + datetime.timedelta(minutes=_i) for _i in range(_total_minutes + 1)
    ]

    # Create a Skyfield time array from the sample datetimes.
    _ts_array = ts.utc(
        [dt.year for dt in _sample_datetimes],
        [dt.month for dt in _sample_datetimes],
        [dt.day for dt in _sample_datetimes],
        [dt.hour for dt in _sample_datetimes],
        [dt.minute for dt in _sample_datetimes],
        [dt.second for dt in _sample_datetimes],
    )

    # Define an altitude threshold (in degrees) for a satellite to be considered "visible".
    threshold_alt_deg = 15

    visible_data = []
    for _s in satellites_sky:
        print(_s)
        # Compute the satellite's apparent position as seen by the observer at each sample time.
        _difference = (_s - observer).at(_ts_array)
        _alt, _az, _distance = _difference.altaz()
        _altitudes = _alt.degrees
        # Count the number of minutes the satellite's altitude exceeds the threshold.
        _visible_minutes = int(np.sum(_altitudes > threshold_alt_deg))
        visible_data.append(
            {"OBJECT_NAME": _s.name, "visible_minutes": _visible_minutes}
        )
    return observer, satellites_sky, threshold_alt_deg, visible_data


@app.cell
def _(mo, pl, visible_data):
    # Create a Polars DataFrame summarizing the visible time (in minutes) for each satellite.
    df_visible = pl.DataFrame(visible_data)
    table_visible = mo.ui.table(df_visible, selection="single", page_size=20)
    table_visible
    return df_visible, table_visible


@app.cell
def _(
    datetime,
    end_date,
    mo,
    np,
    observer,
    pl,
    satellites_sky,
    sensor_lat,
    sensor_lon,
    sensor_name,
    start_date,
    table_visible,
    threshold_alt_deg,
    ts,
):
    _selected = table_visible.value
    mo.stop(_selected.is_empty())
    _sat_data = _selected[0]
    # Ensure _selected_sat_name is a string, not a Polars Series
    _selected_sat_name = _sat_data["OBJECT_NAME"]
    if isinstance(_selected_sat_name, pl.Series):
        _selected_sat_name = _selected_sat_name.item(0)
    else:
        _selected_sat_name = str(_selected_sat_name)

    # Find the corresponding satellite object in satellites_sky
    _selected_sat = None
    for _s in satellites_sky:
        if _s.name == _selected_sat_name:
            _selected_sat = _s
            break
    mo.stop(_selected_sat is None)

    # Define the propagation interval using user-provided start_date and end_date
    _t0 = ts.utc(
        start_date.value.year,
        start_date.value.month,
        start_date.value.day,
        0,
        0,
        0,
    )
    _t1 = ts.utc(
        end_date.value.year, end_date.value.month, end_date.value.day, 23, 59, 59
    )
    _t0_dt = _t0.utc_datetime()
    _t1_dt = _t1.utc_datetime()
    _total_minutes = int((_t1_dt - _t0_dt).total_seconds() // 60)
    _sample_datetimes = [
        _t0_dt + datetime.timedelta(minutes=_i) for _i in range(_total_minutes + 1)
    ]
    _ts_array = ts.utc(
        [dt.year for dt in _sample_datetimes],
        [dt.month for dt in _sample_datetimes],
        [dt.day for dt in _sample_datetimes],
        [dt.hour for dt in _sample_datetimes],
        [dt.minute for dt in _sample_datetimes],
        [dt.second for dt in _sample_datetimes],
    )

    # Propagate the selected satellite and compute its ground track (subpoint positions)
    _subpoints = _selected_sat.at(_ts_array).subpoint()
    _lats = _subpoints.latitude.degrees
    _lons = _subpoints.longitude.degrees

    # Build a Polars DataFrame for the full ground track
    _ground_track = []
    for _dt, _lat, _lon in zip(_sample_datetimes, _lats, _lons):
        _ground_track.append({"time": _dt, "lat": _lat, "lon": _lon})
    _df_ground_track = pl.DataFrame(_ground_track)

    # Compute the apparent altitude of the satellite as seen by the observer
    _difference = (_selected_sat - observer).at(_ts_array)
    _alt, _az, _distance = _difference.altaz()
    _alt_degrees = _alt.degrees

    # Define the altitude threshold (in degrees)

    # Identify indices where the satellite's altitude exceeds (visible) or does not exceed (invisible) the threshold
    _visible_indices = np.where(_alt_degrees > threshold_alt_deg)[0]
    _invisible_indices = np.where(_alt_degrees <= threshold_alt_deg)[0]

    # Build DataFrames for visible ground track points
    _visible_ground_track = []
    for _i in _visible_indices:
        _visible_ground_track.append(
            {"time": _sample_datetimes[_i], "lat": _lats[_i], "lon": _lons[_i]}
        )
    _df_visible_ground_track = pl.DataFrame(_visible_ground_track)

    # Build DataFrames for invisible ground track points
    _invisible_ground_track = []
    for _i in _invisible_indices:
        _invisible_ground_track.append(
            {"time": _sample_datetimes[_i], "lat": _lats[_i], "lon": _lons[_i]}
        )
    _df_invisible_ground_track = pl.DataFrame(_invisible_ground_track)

    # Use vega_datasets (aliased as _vega_data) and Altair (aliased as alt2) for plotting.
    from vega_datasets import data as _vega_data
    import altair as alt2

    _countries = alt2.topo_feature(_vega_data.world_110m.url, "countries")
    _background = (
        alt2.Chart(_countries)
        .mark_geoshape(fill="lightgray", stroke="white")
        .properties(width=1200, height=600)
        .project(
            type="equalEarth",
            rotate=[-float(sensor_lon.value), -float(sensor_lat.value), 0],
        )
    )

    # Plot visible ground track points as green dots.
    _visible_chart = (
        alt2.Chart(_df_visible_ground_track.to_pandas())
        .mark_circle(size=5, color="green")
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            tooltip=[alt2.Tooltip("time:T", title="Time")],
        )
    )

    # Plot invisible ground track points as red dots.
    _invisible_chart = (
        alt2.Chart(_df_invisible_ground_track.to_pandas())
        .mark_circle(size=5, color="red")
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            tooltip=[alt2.Tooltip("time:T", title="Time")],
        )
    )

    # Build a blue dot for the sensor site.
    import pandas as pd

    _site_data = [
        {
            "lat": sensor_lat.value,
            "lon": sensor_lon.value,
            "name": sensor_name.value,
        }
    ]
    _site_df = pd.DataFrame(_site_data)
    _site_chart = (
        alt2.Chart(_site_df)
        .mark_circle(size=100, color="blue")
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            tooltip=[alt2.Tooltip("name:N", title="Site")],
        )
    )

    _chart = _background + _visible_chart + _invisible_chart + _site_chart
    mo.ui.altair_chart(_chart)
    _chart
    return alt2, pd


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
