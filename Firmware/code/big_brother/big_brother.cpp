#include <signal.h>
#include <iostream>
#include "big_brother_state_machine.h"
#include <condition_variable>
#include <atomic>
#include <mutex>

std::mutex button_press_mutex;
std::condition_variable button_press_cv;
std::atomic_bool is_button_pressed;

BigBrotherStateMachine state_machine;

void button_pressed(int) 
{
    is_button_pressed.store(true, std::memory_order::memory_order_seq_cst);
    button_press_cv.notify_all();
}

void callback_init(BigBrotherStateMachine::State)
{
    std::cout << "Callback: INIT" << '\n';
}

void callback_stage_1(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_1" << '\n';

    // !TEMP
    state_machine.stage2_complete();
}

void callback_stage_2(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_2" << '\n';
}

void callback_stage_3(BigBrotherStateMachine::State)
{
    std::cout << "Callback: STAGE_3" << '\n';
}


int main(int, char*[])
{
    // Register the callbacks for each state
    state_machine.register_callback(BigBrotherStateMachine::State::INIT, callback_init);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_1, callback_stage_1);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_2, callback_stage_2);
    state_machine.register_callback(BigBrotherStateMachine::State::STAGE_3, callback_stage_3);

    signal(SIGINT, button_pressed);

    while (1)
    {
        auto lock = std::unique_lock<std::mutex>(button_press_mutex);
        button_press_cv.wait(lock, [&](){ return is_button_pressed.load(std::memory_order_seq_cst); });

        state_machine.button_pressed();
        is_button_pressed.store(false, std::memory_order_seq_cst);
    }

    return 0;
}