#include <signal.h>
#include <iostream>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>

#include "band_buddy_msg.h"
#include "band_buddy_server.h"
#include "big_brother_state_machine.h"

// Mutex for button presses.
std::mutex button_press_mutex;
// Condition variable for button presses.
std::condition_variable button_press_cv;
// Spurious wakeup guard for button presses.
std::atomic_bool is_button_pressed;

// The state machine that backs Big Brother.
BigBrotherStateMachine state_machine;

// Network backbone file descriptor
int networkbb_fd;

// Button pressed triggers on SIGINT
void button_pressed(int) 
{
    is_button_pressed.store(true, std::memory_order::memory_order_seq_cst);
    button_press_cv.notify_all();
}

// Should be run asynchronously - awaits a Stage 2 complete message from the network backbone
void await_stage2_done()
{
    int stage = BIG_BROTHER;
    int ret = FAILED;
    if ((ret = recieve_ack(networkbb_fd, stage)) == FAILED)
    {
        std::cerr << "Receiving Stage 2 completion ACK from the network backbone failed!" << '\n';
        exit(1);
    }
    
    // That's all 
    std::cout << "Stage 2 complete!" << '\n';
    state_machine.stage2_complete();
}


int await_stage_ack()
{
    int stage = BIG_BROTHER;
    int ret = FAILED;
    if ((ret = recieve_ack(networkbb_fd, stage)) == FAILED)
    {
        std::cerr << "Receiving stage ACK from network backbone failed!" << '\n';
        exit(1);
    }

    return ret;
}

// State INIT callback.
void callback_init(BigBrotherStateMachine::State)
{
    std::cout << "Callback: INIT" << '\n';

    // Start Stage 1 recording
    int stage = BIG_BROTHER;
    stage1_start(networkbb_fd, stage);
    
    // Wait for Stage 1 ACK
    await_stage_ack();
}

// State STAGE_1 callback.
void callback_stage_1(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_1" << '\n';

    // Stop Stage 1 recording
    int stage = BIG_BROTHER;
    stage1_stop(networkbb_fd, stage);

    // Wait for Stage 1 ACK
    await_stage_ack();

    // Asynchronously wait for Stage 2 to complete 
    auto stage2_thread = std::thread(await_stage2_done);
    stage2_thread.detach();
}

// State STAGE_2 callback.
void callback_stage_2(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_2" << '\n';

    // Ignore button presses for Stage 2
}

// State STAGE_3 callback.
void callback_stage_3(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_3" << '\n';

    // Stop Stage 3 playback 
    int stage = BIG_BROTHER;
    stage3_stop(networkbb_fd, stage);

    // Wait for Stage 3 ACK
    await_stage_ack();
}


int main(int, char*[])
{
    // Register the callbacks for each state
    state_machine.register_callback(BigBrotherStateMachine::State::INIT, callback_init);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_1, callback_stage_1);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_2, callback_stage_2);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_3, callback_stage_3);

    // Register with the bb
    int stage = BIG_BROTHER;
    if (connect_and_register(stage, networkbb_fd) != SUCCESS)
    {
        std::cerr << "Could not connect and register with the network backbone!" << '\n';
        return 1;
    }

    signal(SIGINT, button_pressed);


    while (1)
    {
        // Await a button press
        auto lock = std::unique_lock<std::mutex>(button_press_mutex);
        button_press_cv.wait(lock, [&](){ return is_button_pressed.load(std::memory_order_seq_cst); });

        // Ask the state machine what to do 
        state_machine.button_pressed();

        // Reset the spurious wakeup guard
        is_button_pressed.store(false, std::memory_order_seq_cst);
    }

    return 0;
}