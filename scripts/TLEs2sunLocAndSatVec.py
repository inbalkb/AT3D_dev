import numpy as np
from skyfield.api import load, wgs84, EarthSatellite, Topos
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

def main():
    # Specify the file name
    filename_tles = "/wdata_visl/inbalkom/NN_Data/EROSB2018TLEs.txt"

    # Read the file content as a single string
    with open(filename_tles, "r") as file:
        tle_data = file.read()

    lines = tle_data.strip().splitlines()

    # Group every two lines into a tuple and store them in a list
    TLEs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    # Display the parsed TLEs
    # for tle in TLEs[:2]:
    #     print(tle)

    # Parse each TLE to get the epoch time and convert to (timestamp, tle_line1, tle_line2) list
    TLEs = [(parse_tle_timestamp(tle1), tle1, tle2) for tle1, tle2 in TLEs]

    # Load ephemeris data and timescale
    ts = load.timescale()
    planets = load("de421.bsp")
    earth, sun = planets["earth"], planets["sun"]

    # Define the time range for one year starting from the first TLE
    start_time = datetime(2018, 1, 1, 3, 51, 0)  # Use a standard datetime object here
    end_time = start_time + timedelta(days=365)
    current_time = start_time
    results = []

    num_days_interval = 1
    minutes_delta = 5
    interval_minutes_range = np.arange(0, num_days_interval * 24 * 60, minutes_delta)  # Total minutes in an interval

    # Loop through time for one year, switching to the closest TLE every 2 days
    while current_time < end_time:
        # Find the closest TLE at or before the current time
        tle_epoch, tle_line1, tle_line2 = find_closest_tle(TLEs, current_time)
        satellite = EarthSatellite(tle_line1, tle_line2, "Satellite", ts)

        # Calculate for 2 days with 1-minute intervals, starting from the last calculated time
        next_time = current_time + timedelta(days=num_days_interval)


        # Generate times in 1-minute intervals using datetime attributes
        times = [current_time + timedelta(minutes=int(i)) for i in interval_minutes_range]
        times = ts.utc([t.year for t in times], [t.month for t in times], [t.day for t in times],
                       [t.hour for t in times], [t.minute for t in times])

        for time in times:
            # Satellite position at this time
            geocentric = satellite.at(time)

            # Get sub-point (latitude, longitude) on Earth directly below the satellite (nadir)
            subpoint = geocentric.subpoint()
            ground_location = Topos(latitude_degrees=subpoint.latitude.degrees,
                                    longitude_degrees=subpoint.longitude.degrees)

            # Calculate Sun azimuth and elevation as seen from the ground location
            astrometric = (earth + ground_location).at(time).observe(sun)
            alt, az, _ = astrometric.apparent().altaz()

            # calculate the satellite's instantaneous direction of motion relative to Earth's north
            motion_direction, lat, lon = get_direction_from_tle(tle_line1, tle_line2, timestamp=time.utc_datetime())

            # Store the result
            if alt.degrees > 0 and (subpoint.latitude.degrees >= -70 and subpoint.latitude.degrees <= 70):
                results.append({
                    "utc_time": time.utc_iso(),
                    "latitude": subpoint.latitude.degrees,
                    "longitude": subpoint.longitude.degrees,
                    "sun_azimuth": az.degrees,
                    "sun_elevation": alt.degrees,
                    "motion_direction": motion_direction,
                })

        # Update to the end of the current interval (2 days later)
        current_time = next_time

    # Display or further process `results` as needed
    for result in results[:10]:  # Display only the first 10 results for brevity
        print(f"Time (UTC): {result['utc_time']}")
        print(f"Location (Lat, Long): ({result['latitude']}, {result['longitude']})")
        print(f"Sun Azimuth: {result['sun_azimuth']:.2f}°, Elevation: {result['sun_elevation']:.2f}°")
        print(f"Motion Direction: {result['motion_direction']:.2f}°")
        print("------------------------------------------------------------")

    # Prepare data for the plots
    azimuths = [result["sun_azimuth"] for result in results]
    elevations = [result["sun_elevation"] for result in results]
    latitudes = [result["latitude"] for result in results]
    times = [result["utc_time"] for result in results]

    # Convert times to numeric values for plotting (e.g., minutes from start)
    time_numeric = np.arange(len(times))*minutes_delta

    # 1. 2D Histogram: Sun's Azimuth vs. Elevation (Heatmap)
    plt.figure(figsize=(10, 6))
    plt.hist2d(azimuths, elevations, bins=30, cmap='hot')
    plt.colorbar(label="Frequency")
    plt.xlabel("Sun's Azimuth (degrees)")
    plt.ylabel("Sun's Elevation (degrees)")
    plt.title("2D Histogram of Sun's Azimuth vs. Elevation")
    plt.show()

    # Specify the filename for the pickle file
    filename = "/wdata_visl/inbalkom/NN_Data/sunsync_satellite_EROSB_sun_angles_and_sat_dir.pkl"

    # Save the results dictionary to a pickle file
    with open(filename, "wb") as file:
        pickle.dump(results, file)

    print(f"Results have been saved to {filename}")


# Function to find the closest TLE that is at or before the target time
def find_closest_tle(TLEs, target_time):
    available_tles = [tle for tle in TLEs if tle[0] <= target_time]

    # Use the closest TLE before or equal to target_time; if none, use the first TLE
    if available_tles:
        closest_tle = max(available_tles, key=lambda x: x[0])
    else:
        closest_tle = TLEs[0]  # Default to the first TLE if no TLEs are earlier than the target time

    return closest_tle[0], closest_tle[1], closest_tle[2]


def parse_tle_timestamp(tle_line1):
    # Extract YYDDD.DDDDDDDD format
    year = int(tle_line1[18:20])
    day_of_year = int(tle_line1[20:23])
    fraction_of_day = float("0" + tle_line1[23:32])

    # Handle the year by converting to full year (assuming year 2000+)
    year += 2000 if year < 57 else 1900  # Adjusting TLE epochs (57 is an arbitrary cutoff for 1957)

    # Calculate the exact timestamp
    base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    timestamp = base_date + timedelta(days=fraction_of_day)
    return timestamp


# def get_direction_from_tle(tle_line1, tle_line2, timestamp=None):
#     """
#     Calculate instantaneous direction of satellite motion relative to Earth's north
#     from TLE data at a given timestamp, using Skyfield.
#
#     Args:
#         tle_line1 (str): First line of TLE
#         tle_line2 (str): Second line of TLE
#         timestamp (datetime, optional): Time at which to calculate direction.
#                                      If None, uses current time.
#
#     Returns:
#         float: Angle in degrees from north (0-360).
#               0/360° = northward
#               90° = eastward
#               180° = southward
#               270° = westward
#     """
#     # Initialize time scales
#     ts = load.timescale()
#
#     # Create satellite object
#     satellite = EarthSatellite(tle_line1, tle_line2, name='SAT', ts=ts)
#
#     # Use current time if none provided
#     if timestamp is None:
#         timestamp = datetime.utcnow()
#
#     # Convert to Skyfield time
#     t = ts.from_datetime(timestamp)
#
#     # Get geocentric position and velocity
#     geocentric = satellite.at(t)
#     pos = geocentric.position.km
#     vel = geocentric.velocity.km_per_s
#
#     # Convert to numpy arrays and normalize
#     pos = np.array(pos)
#     vel = np.array(vel)
#     pos_normalized = pos / np.linalg.norm(pos)
#     vel_normalized = vel / np.linalg.norm(vel)
#
#     # Project velocity onto Earth's surface (remove radial component)
#     vel_projected = vel_normalized - np.dot(vel_normalized, pos_normalized) * pos_normalized
#     vel_projected = vel_projected / np.linalg.norm(vel_projected)
#
#     # Calculate local north vector
#     z_earth = np.array([0, 0, 1])
#     north = z_earth - np.dot(z_earth, pos_normalized) * pos_normalized
#     north = north / np.linalg.norm(north)
#
#     # Calculate angle between projected velocity and north
#     angle = np.arccos(np.clip(np.dot(vel_projected, north), -1.0, 1.0))
#     angle_deg = np.degrees(angle)
#
#     # Determine sign of angle (east vs west of north)
#     east = np.cross(pos_normalized, north)
#     if np.dot(vel_projected, east) < 0:
#         angle_deg = 360 - angle_deg
#
#     return angle_deg

def get_direction_from_tle(tle_line1, tle_line2, timestamp):
    """
    Calculate instantaneous direction and position from TLE data.
    Added position output for verification.
    """
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, name='SAT', ts=ts)

    # Convert to Skyfield time
    if isinstance(timestamp, datetime):
        t = ts.from_datetime(timestamp)
    else:
        t = timestamp

    # Get geocentric position and velocity
    geocentric = satellite.at(t)

    # Get lat/lon for verification
    subpoint = geocentric.subpoint()
    lat = subpoint.latitude.degrees
    lon = subpoint.longitude.degrees

    pos = geocentric.position.km
    vel = geocentric.velocity.km_per_s

    # Convert to numpy arrays
    pos = np.array(pos)
    vel = np.array(vel)

    # Calculate ECEF velocity direction
    vel_normalized = vel / np.linalg.norm(vel)
    pos_normalized = pos / np.linalg.norm(pos)

    # Get local vertical (radial) vector
    radial = pos_normalized

    # Get local north vector (perpendicular to position, in plane with Z axis)
    z_earth = np.array([0, 0, 1])
    north = z_earth - np.dot(z_earth, radial) * radial
    north = north / np.linalg.norm(north)

    # Get local east vector
    east = np.cross(north, radial)

    # Project velocity onto local horizontal plane
    vel_projected = vel_normalized - np.dot(vel_normalized, radial) * radial
    vel_projected = vel_projected / np.linalg.norm(vel_projected)

    # Calculate angle from north using atan2 for correct quadrant
    east_component = np.dot(vel_projected, east)
    north_component = np.dot(vel_projected, north)
    angle_deg = np.degrees(np.arctan2(east_component, north_component)) % 360

    return angle_deg, lat, lon


if __name__ == '__main__':
    main()
