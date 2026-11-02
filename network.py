(f"
python\n{factorial_code}
")
    else:
        edq.display_response("Decision was not to generate factorial function this time.")


    # Cycle 3: More complex instruction with high entropy modulation
    print("\n[Cycle 3: Exploring a novel solution path]")
    sensor_data_3 = get_sensor_vector(num_sensors=NUMERIC_DIM) * 2.0 # Even more different
    text_command_3 = "Propose an alternative data structure for linked lists"
    print(f"  - Sensor Input: {np.round(sensor_data_3, 2)}")
    print(f"  - Text Command: '{text_command_3}'")
    ectar.process_inputs_and_decide(sensor_data_3, text_command_3)
    ectar.display_response("Considering novel data structures. The quantum-inspired entropy seems to be encouraging exploration here.")

    print("\n--- edq System Simulation Ended ---")
