import marimo

__generated_with = "0.11.4"
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
    _start_date = mo.ui.date(
        value=datetime.datetime.now().date(), label="Start Date"
    )

    _end_date = mo.ui.date(value=datetime.datetime.now().date(), label="End Date")

    # Kihei, Maui approximate coordinates:
    _sensor_lat = mo.ui.number(value=20.7649, label="Sensor Latitude (°)")
    _sensor_lon = mo.ui.number(value=-156.3311, label="Sensor Longitude (°)")
    _sensor_alt = mo.ui.number(
        value=0.1,
        label="Sensor Altitude (km)",
    )

    sensor_name = mo.ui.text(
        value="My Sensor", placeholder="Enter sensor name", label="Sensor Name"
    )

    user_input = mo.md(
        "{start_date} {end_date}\n\n{sensor_lat} {sensor_lon} {sensor_alt}"
    ).batch(
        start_date=_start_date,
        end_date=_end_date,
        sensor_lat=_sensor_lat,
        sensor_lon=_sensor_lon,
        sensor_alt=_sensor_alt,
    ).form()

    # user_date = mo.hstack([start_date, end_date], justify="start")
    # user_site = mo.hstack([sensor_lat, sensor_lon, sensor_alt], justify="start")
    # user_input = mo.vstack([user_date, user_site])

    user_input
    return sensor_name, user_input


@app.cell
def _(
    E,
    EarthSatellite,
    N,
    S,
    W,
    datetime,
    df_tles,
    load,
    mo,
    np,
    ts,
    user_input,
    wgs84,
):
    mo.stop(not user_input.value)

    # Load Skyfield timescale
    _planets = load("de421.bsp")
    _earth = _planets["earth"]

    start_date = user_input.value["start_date"]
    end_date = user_input.value["end_date"]
    sensor_lat = user_input.value["sensor_lat"]
    sensor_lon = user_input.value["sensor_lon"]
    sensor_alt = user_input.value["sensor_alt"]

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
        start_date.year,
        start_date.month,
        start_date.day,
        0,
        0,
        0,
    )
    _t1 = ts.utc(
        end_date.year, end_date.month, end_date.day, 23, 59, 59
    )

    # Create an observer location using wgs84.latlon and the sensor inputs.
    # Use N if latitude is positive, S if negative; W if longitude is negative, E if positive.
    observer = wgs84.latlon(
        sensor_lat * (N if sensor_lat >= 0 else S),
        abs(sensor_lon) * (W if sensor_lon < 0 else E),
        elevation_m=sensor_alt
        * 1000,  # sensor_alt is in km, convert to meters
    )

    # Generate time samples at 1-minute intervals between _t0 and _t1.
    _t0_dt = _t0.utc_datetime()
    _t1_dt = _t1.utc_datetime()
    _total_minutes = int((_t1_dt - _t0_dt).total_seconds() // 60)
    sample_datetimes = [
        _t0_dt + datetime.timedelta(minutes=_i) for _i in range(_total_minutes + 1)
    ]

    # Create a Skyfield time array from the sample datetimes.
    _ts_array = ts.utc(
        [dt.year for dt in sample_datetimes],
        [dt.month for dt in sample_datetimes],
        [dt.day for dt in sample_datetimes],
        [dt.hour for dt in sample_datetimes],
        [dt.minute for dt in sample_datetimes],
        [dt.second for dt in sample_datetimes],
    )

    # Define an altitude threshold (in degrees) for a satellite to be considered "visible".
    threshold_alt_deg = 15

    visible_data = []
    with mo.status.progress_bar(satellites_sky) as bar:
        for _s in satellites_sky:
            bar.update(title="Computing visibility", subtitle=f"Satellite: {_s.name}")
            # Compute the satellite's apparent position as seen by the observer at each sample time.
            _difference = (_s - observer).at(_ts_array)
            _alt, _az, _distance = _difference.altaz()
            _altitudes = _alt.degrees
            # Count the number of minutes the satellite's altitude exceeds the threshold.
            _visible_minutes = int(np.sum(_altitudes > threshold_alt_deg))
            visible_data.append(
                {"OBJECT_NAME": _s.name, "visible_minutes": _visible_minutes}
            )
        bar.clear()
    return (
        bar,
        end_date,
        observer,
        sample_datetimes,
        satellites_sky,
        sensor_alt,
        sensor_lat,
        sensor_lon,
        start_date,
        threshold_alt_deg,
        visible_data,
    )


@app.cell
def _(mo, pl, visible_data):
    # Create a Polars DataFrame summarizing the visible time (in minutes) for each satellite.
    df_visible = pl.DataFrame(visible_data)
    table_visible = mo.ui.table(df_visible, selection="single", page_size=20)
    table_visible
    return df_visible, table_visible


@app.cell
def _(datetime, end_date, mo, start_date, ts):
    # Define the propagation interval using user-provided start_date and end_date
    _t0 = ts.utc(
        start_date.year,
        start_date.month,
        start_date.day,
        0,
        0,
        0,
    )
    _t1 = ts.utc(
        end_date.year, end_date.month, end_date.day, 23, 59, 59
    )
    _t0_dt = _t0.utc_datetime()
    _t1_dt = _t1.utc_datetime()
    _total_minutes = int((_t1_dt - _t0_dt).total_seconds() // 60)
    sample_datetimes_ = [
        _t0_dt + datetime.timedelta(minutes=_i) for _i in range(_total_minutes + 1)
    ]
    _now = datetime.datetime.now(datetime.timezone.utc)

    # Find the index in sample_datetimes closest to now.
    default_index = min(
        range(len(sample_datetimes_)),
        key=lambda i: abs(sample_datetimes_[i] - _now),
    )

    # Create a slider to let the user select a time sample index.
    # (The slider returns a numeric value corresponding to an index in sample_datetimes.)
    mo_slider_time_index = mo.ui.slider(
        start=0,
        stop=len(sample_datetimes_) - 1,
        step=10,
        value=default_index,
        label="Select Time",
        show_value=False,
    )
    return default_index, mo_slider_time_index, sample_datetimes_


@app.cell
def _(mo, mo_slider_time_index, sample_datetimes_):
    selected_dt_z = (
        sample_datetimes_[mo_slider_time_index.value]
        .isoformat()
        .replace("+00:00", "Z")
    )
    mo.hstack([mo_slider_time_index, mo.md(f"{selected_dt_z}")], justify="start")
    return (selected_dt_z,)


@app.cell
def _(
    Topos,
    datetime,
    end_date,
    load,
    mo,
    mo_slider_time_index,
    np,
    observer,
    pl,
    satellites_sky,
    sensor_alt,
    sensor_lat,
    sensor_lon,
    sensor_name,
    start_date,
    table_visible,
    threshold_alt_deg,
    ts,
):
    import pandas as pd

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
        start_date.year,
        start_date.month,
        start_date.day,
        0,
        0,
        0,
    )
    _t1 = ts.utc(
        end_date.year, end_date.month, end_date.day, 23, 59, 59
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

    # Define the altitude threshold (in degrees) for the satellite to be visible.
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

    # ----- Compute the Moon separation constraint -----
    # Load an ephemeris and get the Moon object (using Skyfield’s de421 for example)
    _eph = load("de421.bsp")
    _earth = _eph['earth']
    _moon = _eph['moon']

    _observer = _earth + Topos(
        latitude_degrees=sensor_lat,
        longitude_degrees=sensor_lon,
        elevation_m=sensor_alt * 1000,  # sensor_alt is in km, convert to meters
    )

    # Compute the Moon's apparent position as seen by the observer at all sample times
    _moon_difference = _observer.at(_ts_array).observe(_moon)

    # Calculate the angular separation (in degrees) between the satellite and the Moon
    _moon_sep_degrees = _difference.separation_from(_moon_difference).degrees

    # Define a moon separation threshold (in degrees) below which the satellite is considered blocked by the Moon.
    threshold_moon_sep_deg = 30  # Adjust as needed

    # Identify indices where the Moon separation is less than the threshold.
    _moon_blocked_indices = np.where(_moon_sep_degrees < threshold_moon_sep_deg)[0]

    # Build DataFrame for ground track points where the Moon separation constraint is violated.
    _moon_blocked_ground_track = []
    for _i in _moon_blocked_indices:
        _moon_blocked_ground_track.append(
            {"time": _sample_datetimes[_i], "lat": _lats[_i], "lon": _lons[_i]}
        )
    _df_moon_blocked_ground_track = pl.DataFrame(_moon_blocked_ground_track)
    # ----- End Moon Separation Section -----

    # ----- Compute the AtNightConstraint -----
    # Compute the Sun's apparent position as seen by the observer at all sample times.
    _sun = _eph['sun']
    _sun_difference = _observer.at(_ts_array).observe(_sun).apparent()
    _sun_alt, _sun_az, _sun_distance = _sun_difference.altaz()
    _sun_alt_degrees = _sun_alt.degrees

    # Define a threshold for darkness. In this example, the sensor is considered dark enough 
    # if the Sun is below -6° (civil twilight). Adjust this value as needed.
    threshold_sun_alt_deg = -6

    # Identify indices where it is daytime (i.e. Sun altitude is greater than or equal to the threshold).
    _daytime_indices = np.where(_sun_alt_degrees >= threshold_sun_alt_deg)[0]

    # Build DataFrame for ground track points where it is daytime.
    _daytime_ground_track = []
    for _i in _daytime_indices:
        _daytime_ground_track.append(
            {"time": _sample_datetimes[_i], "lat": _lats[_i], "lon": _lons[_i]}
        )
    _df_daytime_ground_track = pl.DataFrame(_daytime_ground_track)
    # ----- End AtNightConstraint Section -----

    # ---------------------------
    # Add a marker along the satellite track for a selected time.
    # By default, we use the current UTC time.
    # Convert current UTC time to a Python datetime object.
    # _now = datetime.datetime.now(datetime.timezone.utc)

    # # Find the index in _sample_datetimes closest to now.
    # default_index = min(range(len(_sample_datetimes)), key=lambda i: abs(_sample_datetimes[i] - _now))

    # Create a slider to let the user select a time sample index.
    # (The slider returns a numeric value corresponding to an index in _sample_datetimes.)
    # selected_index = mo.ui.slider(
    #     start=0,
    #     stop=len(_sample_datetimes) - 1,
    #     step=1,
    #     value=default_index,
    #     label="Select Time Index",
    #     show_value=True
    # )
    selected_index = mo_slider_time_index.value

    # Build a DataFrame for the marker position on the satellite ground track
    _marker_data = [{
        "time": _sample_datetimes[selected_index],
        "lat": _lats[selected_index],
        "lon": _lons[selected_index]
    }]
    _df_marker = pd.DataFrame(_marker_data)

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
            rotate=[-float(sensor_lon), -float(sensor_lat), 0],
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

    # Plot Moon-blocked ground track points as orange dots.
    _moon_blocked_chart = (
        alt2.Chart(_df_moon_blocked_ground_track.to_pandas())
        .mark_circle(size=5, color="purple")
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            tooltip=[alt2.Tooltip("time:T", title="Time (Moon Blocked)")],
        )
    )

    # Plot daytime ground track points as yellow dots.
    _daytime_chart = (
        alt2.Chart(_df_daytime_ground_track.to_pandas())
        .mark_circle(size=5, color="yellow")
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            tooltip=[alt2.Tooltip("time:T", title="Time (Daytime)")],
        )
    )

    # Build a blue dot for the sensor site.
    _site_data = [
        {
            "lat": sensor_lat,
            "lon": sensor_lon,
            "name": sensor_name,
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

    # Create a marker chart for the selected time (using a magenta dot)
    _marker_chart = alt2.Chart(_df_marker).mark_point(size=50, color="magenta", shape="diamond",opacity=1).encode(
        longitude="lon:Q",
        latitude="lat:Q",
        tooltip=[alt2.Tooltip("time:T", title="Selected Time")]
    )
    # ---------------------------

    # Combine all layers: background, visible, invisible, moon-blocked, daytime, and sensor site.
    _chart = (
        _background
        + _visible_chart
        + _invisible_chart
        + _moon_blocked_chart
        + _daytime_chart
        + _site_chart
        + _marker_chart
    )
    mo.ui.altair_chart(_chart)
    return (
        alt2,
        pd,
        selected_index,
        threshold_moon_sep_deg,
        threshold_sun_alt_deg,
    )


if __name__ == "__main__":
    app.run()
