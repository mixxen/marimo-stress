import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime
    import httpx

    import altair as alt
    import polars as pl
    import marimo as mo
    import numpy as np

    from vega_datasets import data as vega_data

    from sgp4.api import Satrec, jday
    from sgp4.api import WGS72

    return Satrec, WGS72, alt, datetime, httpx, jday, mo, np, pl, vega_data


@app.cell
def _(mo):
    mo_btn_refresh = mo.ui.run_button(
        label="Refresh Data",
        tooltip="Fetch the latest data.",
    )
    mo.hstack([mo_btn_refresh], justify="end")
    return (mo_btn_refresh,)


@app.cell
def _(httpx, mo_btn_refresh, pl):
    def fetch_gp_data(url: str):
        """
        Fetch data from the given URL using httpx.
        Raises an error if the HTTP response status is not 200.
        Returns the JSON-decoded response.
        """
        response = httpx.get(url)
        response.raise_for_status()  # Will raise an error for non-200 responses.
        return response.json()

    # refresh the data when the button is clicked
    mo_btn_refresh

    # Define the CelesTrak GP query URL.
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=ACTIVE&FORMAT=JSON"

    # Fetch the GP data from CelesTrak.
    data = fetch_gp_data(url)

    # Depending on the JSON structure, data might be a list of records,
    # or a dictionary wrapping the data. Adjust accordingly:
    if isinstance(data, dict):
        # If the JSON contains a top-level key (e.g., "GP"), use that.
        if "GP" in data:
            data = data["GP"]
        else:
            # If it’s a single record, wrap it in a list.
            data = [data]

    # Create a Polars DataFrame from the JSON data.
    df = pl.DataFrame(data)
    return data, df, fetch_gp_data, url


@app.cell
def _(df, mo):
    table = mo.ui.table(df, selection="single")
    table
    return (table,)


@app.cell
def _(Satrec, WGS72, datetime, jday, mo, np, table):
    # Retrieve the selected rows from the GP data table.
    selected = table.value
    mo.stop(selected.is_empty())

    sat_data = selected[0]
    print("Selected satellite data:", sat_data)

    # Extract the EPOCH value from the Series (assume each field is a Series with one element)
    epoch_str = sat_data["EPOCH"][0]
    print("Epoch string:", epoch_str)

    epoch_dt = datetime.datetime.fromisoformat(epoch_str)
    jd_val, fr_val = jday(
        epoch_dt.year, epoch_dt.month, epoch_dt.day,
        epoch_dt.hour, epoch_dt.minute,
        epoch_dt.second + epoch_dt.microsecond / 1e6
    )
    epoch_arg = jd_val + fr_val - 2433281.5

    # Parse and convert satellite parameters (extracting the first element of each Series)
    sat_num = int(sat_data["NORAD_CAT_ID"][0])
    bstar = float(sat_data["BSTAR"][0])
    ndot = float(sat_data["MEAN_MOTION_DOT"][0])
    nddot = float(sat_data["MEAN_MOTION_DDOT"][0])
    ecc = float(sat_data["ECCENTRICITY"][0])
    argpo = np.radians(float(sat_data["ARG_OF_PERICENTER"][0]))
    inclo = np.radians(float(sat_data["INCLINATION"][0]))
    mo_val = np.radians(float(sat_data["MEAN_ANOMALY"][0]))
    ra = np.radians(float(sat_data["RA_OF_ASC_NODE"][0]))
    mean_motion_rev_day = float(sat_data["MEAN_MOTION"][0])
    no_val = mean_motion_rev_day * 2 * np.pi / 1440.0  # Convert rev/day to rad/min

    # Initialize the satellite using the WGS72 gravity model and improved mode ('i')
    satellite = Satrec()
    satellite.sgp4init(
        WGS72,
        'i',
        sat_num,
        epoch_arg,
        bstar,
        ndot,
        nddot,
        ecc,
        argpo,
        inclo,
        mo_val,
        no_val,
        ra
    )

    return (
        argpo,
        bstar,
        ecc,
        epoch_arg,
        epoch_dt,
        epoch_str,
        fr_val,
        inclo,
        jd_val,
        mean_motion_rev_day,
        mo_val,
        nddot,
        ndot,
        no_val,
        ra,
        sat_data,
        sat_num,
        satellite,
        selected,
    )


@app.cell
def _(Satrec, datetime, jday, mo, np, pl, satellite):
    # # ---------------------------------------------------------------------------
    # # Step 1. (Simulated) "Click" on a satellite by providing its TLE lines.
    # # In your application these might be extracted from the GP data you previously fetched.
    # tle_line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
    # tle_line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

    # # Create a satellite object from the TLE lines.
    # satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    # ---------------------------------------------------------------------------
    # Step 2. Propagate the satellite over the next 24 hours in 1-minute increments.
    def propagate_satellite(sat: Satrec, start_time: datetime) -> pl.DataFrame:
        """
        Propagate the given satellite from start_time over the next 24 hours
        at 1-minute intervals using sgp4_array. Return a Polars DataFrame with
        the time, x, y, z positions (in km) and the error code for each propagation.
        """
        # Generate timestamps for every minute over the next 24 hours (1440 points)
        times = [start_time + datetime.timedelta(minutes=i) for i in range(24 * 60)]

        # Convert each timestamp to a Julian date (jd, fr) using sgp4.api.jday.
        # (The fractional part is computed from seconds + microseconds.)
        jd_list, fr_list = [], []
        for t in times:
            jd, fr = jday(t.year, t.month, t.day,
                          t.hour, t.minute,
                          t.second + t.microsecond / 1e6)
            jd_list.append(jd)
            fr_list.append(fr)

        # Convert lists to numpy arrays for vectorized propagation.
        jd_array = np.array(jd_list)
        fr_array = np.array(fr_list)

        # Use the array version of sgp4 propagation.
        # error: an array of error codes for each time step
        # positions: a (n, 3) array of x,y,z positions in kilometers
        # velocities: a (n, 3) array of velocities in km/s (unused here)
        error, positions, velocities = sat.sgp4_array(jd_array, fr_array)

        # Build a Polars DataFrame from the results.
        # (Positions and error are converted to lists for each column.)
        df = pl.DataFrame({
            "time": times,
            "x": positions[:, 0].tolist(),
            "y": positions[:, 1].tolist(),
            "z": positions[:, 2].tolist(),
            "error": error.tolist()
        })
        return df

    # Choose a starting time for the propagation.
    # (For example, we use the current UTC time.)
    start_time = datetime.datetime.utcnow()

    # Propagate the satellite for the next 24 hours.
    df_positions = propagate_satellite(satellite, start_time)

    # ---------------------------------------------------------------------------
    # Step 3. Display the propagation results using Marimo.
    # The table displays the DataFrame, and 'selection="multi"' allows multiple rows to be selected.
    table_positions = mo.ui.table(df_positions, selection="multi")
    table_positions
    return df_positions, propagate_satellite, start_time, table_positions


@app.cell
def _(alt, df_positions, jday, mo, np, pl, vega_data):
    def teme_to_latlon(x, y, z, dt):
        """
        Convert TEME (x,y,z in km) to geodetic latitude and longitude (in degrees)
        for a given datetime dt.

        This function computes a rough conversion using a basic GMST calculation.
        """
        # Compute the full Julian Date (JD) from the datetime.
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute,
                      dt.second + dt.microsecond / 1e6)
        JD = jd + fr

        # Compute GMST (in hours) using a common approximation.
        # (Reference epoch: J2000.0 at JD 2451545.0)
        GMST_hours = 18.697374558 + 24.06570982441908 * (JD - 2451545.0)
        # Bring into the 0-24 hour range.
        GMST_hours = GMST_hours % 24
        # Convert GMST to radians.
        GMST_rad = GMST_hours * (2 * np.pi / 24)

        # Rotate the TEME x,y coordinates to get approximate ECEF coordinates.
        x_ecef = x * np.cos(GMST_rad) + y * np.sin(GMST_rad)
        y_ecef = -x * np.sin(GMST_rad) + y * np.cos(GMST_rad)
        # z remains the same in this simplified conversion.

        # Compute geodetic latitude and longitude.
        r = np.sqrt(x_ecef**2 + y_ecef**2)
        lat_rad = np.arctan2(z, r)
        lon_rad = np.arctan2(y_ecef, x_ecef)
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        return lat_deg, lon_deg

    # --- Add latitude and longitude columns to df_positions ---
    # Convert the Polars DataFrame to a list of dictionaries for iteration.
    positions_dicts = df_positions.to_dicts()

    lat_list = []
    lon_list = []
    for row in positions_dicts:
        dt = row["time"]  # a datetime object
        x = row["x"]
        y = row["y"]
        z = row["z"]
        lat, lon = teme_to_latlon(x, y, z, dt)
        lat_list.append(lat)
        lon_list.append(lon)

    # Append the computed latitudes and longitudes to df_positions.
    df_positions_latlon = df_positions.with_columns([
        pl.Series("lat", lat_list),
        pl.Series("lon", lon_list)
    ])


    # Load world map data (TopoJSON format) from vega_datasets.
    countries = alt.topo_feature(vega_data.world_110m.url, 'countries')

    # Create a background world map using a geoshape.
    background = alt.Chart(countries).mark_geoshape(
        fill='lightgray',
        stroke='white'
    ).properties(
        width=800,
        height=400
    ).project(
        type='equalEarth'  # using the Equal Earth projection
    )

    # Create a layer for the satellite positions.
    # Convert df_positions_latlon to a pandas DataFrame for Altair.
    points = alt.Chart(df_positions_latlon.to_pandas()).mark_circle(size=30, color='red').encode(
        longitude='lon:Q',
        latitude='lat:Q',
        tooltip=[alt.Tooltip('time:T', title='Time')]
    )

    # Combine the background and point layers.
    chart = background + points

    # Render the chart using Marimo's altair_chart UI function.
    mo.ui.altair_chart(chart)
    return (
        background,
        chart,
        countries,
        df_positions_latlon,
        dt,
        lat,
        lat_list,
        lon,
        lon_list,
        points,
        positions_dicts,
        row,
        teme_to_latlon,
        x,
        y,
        z,
    )


@app.cell
def _(alt, df_positions_latlon, mo, pl):
    # Compute satellite passes over the United States.
    # Define approximate US bounding box: lat between 24° and 50°, lon between -125° and -66°.
    passes_list = []
    in_pass = False
    _start_time = None
    _prev_time = None

    for _row in df_positions_latlon.to_dicts():
        _lat = _row["lat"]
        _lon = _row["lon"]
        current_time = _row["time"]
        # Check if the satellite is over the US.
        if (_lat >= 24) and (_lat <= 50) and (_lon >= -125) and (_lon <= -66):
            if not in_pass:
                in_pass = True
                _start_time = current_time
            _prev_time = current_time
        else:
            if in_pass:
                passes_list.append({
                    "start_time": _start_time,
                    "end_time": _prev_time,
                    "duration": (_prev_time - _start_time).total_seconds() / 60.0
                })
                in_pass = False
                _start_time = None
                _prev_time = None

    # If a pass is still ongoing at the end of the propagation data, record it.
    if in_pass:
        passes_list.append({
            "start_time": _start_time,
            "end_time": _prev_time,
            "duration": (_prev_time - _start_time).total_seconds() / 60.0
        })

    # Add a pass identifier.
    for i, p in enumerate(passes_list):
        p["pass_id"] = f"Pass {i+1}"

    # Create a Polars DataFrame from the passes list.
    df_passes = pl.DataFrame(passes_list)

    # Create a timeline chart for the satellite passes over the US.
    pass_chart = alt.Chart(df_passes.to_pandas()).mark_bar().encode(
        x=alt.X("start_time:T", title="Start Time"),
        x2=alt.X2("end_time:T", title="End Time"),
        y=alt.Y("pass_id:N", title="Pass"),
        tooltip=[
            alt.Tooltip("start_time:T", title="Start"),
            alt.Tooltip("end_time:T", title="End"),
            alt.Tooltip("duration:Q", title="Duration (min)")
        ]
    ).properties(
        height=100,
        title="Satellite Passes over the United States"
    )

    mo.ui.altair_chart(pass_chart)

    return current_time, df_passes, i, in_pass, p, pass_chart, passes_list


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
