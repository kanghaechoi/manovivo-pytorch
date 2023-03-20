class Read:
    def read_txt(self, file_path: str) -> list:
        with open(file_path, "r") as file:
            data = []
            buffer = []

            previous_direction = 0
            current_direction = 0

            swap_count = 0

            for row in file:
                row_data = row.split()
                current_direction = float(row_data[0])

                for row_element in row_data:
                    if abs(current_direction + previous_direction) <= 2:
                        data.append(buffer)

                        previous_direction = current_direction
                        buffer = []
                        # swap_count += 1

                    else:
                        buffer.append(row_element)

                    # if swap_count == 17:
                    #     break

        return data
