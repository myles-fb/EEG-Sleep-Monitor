#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

#include "board_shim.h"
#include "brainflow_input_params.h"
#include "brainflow_array.h"

int main() {
    // 1. Prepare board parameters
    BrainFlowInputParams params;
    params.serial_port = "/dev/ttyUSB0";

    int board_id = (int)BoardIds::CYTON_BOARD;

    // 2. Create board object
    BoardShim board(board_id, params);

    // 3. Prepare session
    board.prepare_session();

    // 4. Start streaming
    board.start_stream();

    // 5. Wait and collect data
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 6. Get data
    BrainFlowArray<double, 2> data = board.get_board_data();

    // 7. Stop and release
    board.stop_stream();
    board.release_session();

    // 8. Print data
    int rows = data.get_size(0);
    int cols = data.get_size(1);

    std::cout << "Data shape: " << rows << " x " << cols << "\n\n";
    std::cout << "Printing first 20 samples:\n";

    for (int c = 0; c < std::min(cols, 20); c++) {
        std::cout << "Sample " << c << ": ";

        for (int r = 0; r < rows; r++) {
            std::cout << data.at(r, c) << " ";
        }

        std::cout << "\n";
    }

    return 0;
}
