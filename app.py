from flask import Flask, render_template, request
import numpy as np
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        krawedz_pokoju = float(request.form['krawedz_pokoju'])
        temperatura_docelowa = float(request.form['temperatura_docelowa'])
        czas_trwania_symulacji = float(request.form['czas_trwania_symulacji'])
        wysokosc = float(request.form['wysokosc'])
        moc_grzejnika = float(request.form['moc_grzejnika'])
        time, room_temperature, heat_given, outside_temperature, heat_loss = automatyka(krawedz_pokoju, temperatura_docelowa,
                                                                             czas_trwania_symulacji, wysokosc,
                                                                             moc_grzejnika)

        fig = go.Figure(data=go.Scatter(x=time, y=room_temperature))
        fig.update_layout(xaxis_title='Czas (minuty)',
                          yaxis_title='Temperatura (C)')
        fig2 = go.Figure(data=go.Scatter(x=time, y=heat_given*60))
        fig2.update_layout(xaxis_title='Czas (minuty)', yaxis_title='Ciepło (J)')
        fig3 = go.Figure(data=go.Scatter(x=time, y=outside_temperature))
        fig3.update_layout(xaxis_title='Czas (minuty)',
                           yaxis_title='Temperatura (C)')
        fig4 = go.Figure(data=go.Scatter(x=time, y=heat_loss*60))
        fig4.update_layout(xaxis_title='Czas (minuty)', yaxis_title='Ciepło (J)')

        return render_template('index.html', fig=fig.to_html(), fig2=fig2.to_html(), fig3=fig3.to_html(), fig4=fig4.to_html())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()


def automatyka(krawedz_pokoju, temperatura_docelowa, czas_trwania_symulacji, wysokosc, moc_grzejnika):
    # user inputs
    room_edge = krawedz_pokoju
    desired_temp = temperatura_docelowa
    simulation_duration = (czas_trwania_symulacji * 60 * 24 * 60)  # user days to simulation minutes
    alitiude = wysokosc  # meters over sea level
    P_heater = moc_grzejnika  # Watts

    # Constants
    outside_temp_amplitude = np.random.uniform(-5, 15)  # Amplitude of outside temperature change
    heat_transfer_coefficient = 0.4  # W/m^2K
    s_heat_capacity = 1012  # J/kgK
    measurement_interval = 60  # seconds

    R_d = 287.058  # J/kgK gas constant for dry air
    R_ideal = 8.314  # J/molK ideal gas constant
    air_pressure_at_sea_level = 1013  # hPa
    air_molar_mass = 0.0289647  # kg/mol
    g_acceleration = 9.80665  # m/s^2

    # Calculated constants
    room_volume = room_edge ** 3
    room_surface_area = 5 * (room_edge ** 2)

    # Calculate number of measurements
    num_measurements = int(simulation_duration / measurement_interval)

    # Initialize arrays to store data
    time = np.arange(0, num_measurements)
    room_temperature = np.zeros(num_measurements)
    heat_given = np.zeros(num_measurements)
    heat_loss = np.zeros(num_measurements)



    # Read the CSV file and extract the desired column
    df = pd.read_csv('data.csv')
    temperature_hours = df.iloc[1:, 1].tolist()

    # Convert temperature values from hours to minutes
    temperature_minutes = temperature_hours

    # Interpolate the temperature values to get minute values between the hourly values
    temperature_interpolated = np.interp(np.arange(len(temperature_minutes) * 60),
                                         np.arange(0, len(temperature_minutes) * 60, 60), temperature_minutes)

    # Add noise to the temperature values
    outside_temperature = [temp + np.random.uniform(-0.1, 0.1) for temp in temperature_interpolated]
    # Initialize room temperature to be the same as outside temperature
    room_temperature[0] = outside_temperature[0]

    # PID gains Ziegler-Nichols method
    Kp = 0.6
    Ki = 0.0008
    Kd = 7.5
    integral = 0
    previous_error = 0
    umax = 1
    umin = 0

    # simulate temperature control
    for i in range(1, num_measurements):

        # Calculate error
        error = desired_temp - room_temperature[i - 1]

        # Proportional term
        P = Kp * error

        # Integral term
        integral += Ki * error * measurement_interval

        # Derivative term
        derivative = (error - previous_error) / measurement_interval

        # PID output
        u = P + integral + (Kd * derivative)

        # Calculate heat given
        heat_given[i] = P_heater * min(max(u, umin), umax)

        # Calculate heat loss
        heat_loss[i] = room_surface_area * heat_transfer_coefficient * (room_temperature[i - 1] - outside_temperature[i - 1])

        # Calculate air pressure
        air_pressure = air_pressure_at_sea_level * np.exp(
            -g_acceleration * air_molar_mass * alitiude / (R_ideal * (room_temperature[i - 1] + 273.15)))

        # Calculate air density
        air_denstiy = (air_pressure / (R_d * (room_temperature[i - 1] + 273.15))) * 100

        # Calculate air mass
        air_mass = air_denstiy * room_volume

        # Calculate room tempreture
        room_temperature[i] = (measurement_interval / (air_mass * s_heat_capacity)) * (
                heat_given[i] - heat_loss[i]) + room_temperature[i - 1]

        # Update previous error
        previous_error = error
    """   # Plot results
    fig = go.Figure(data=go.Scatter(x=time, y=room_temperature))
    fig.update_layout(title='Room temperature over time', xaxis_title='Time (minutes)', yaxis_title='Temperature (C)')
    fig.show()

    fig2 = go.Figure(data=go.Scatter(x=time, y=heat_given))
    fig2.update_layout(title='Heat given over time', xaxis_title='Time (minutes)', yaxis_title='Heat (W)')
    fig2.show()

    fig3 = go.Figure(data=go.Scatter(x=time, y=outside_temperature))
    fig3.update_layout(title='Outside temperature over time', xaxis_title='Time (minutes)', yaxis_title='Temperature (C)')
    fig3.show()
    """
    return time, room_temperature, heat_given, outside_temperature, heat_loss
