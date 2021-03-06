#include "big_brother_state_machine.h"
#include "band_buddy_msg.h"
#include "band_buddy_server.h"

#include <stdexcept>


std::unordered_map<BigBrotherStateMachine::State, BigBrotherStateMachine::State> BigBrotherStateMachine::state_transition_map = 
{
    { BigBrotherStateMachine::State::INIT, BigBrotherStateMachine::State::STAGE_1 },
    { BigBrotherStateMachine::State::STAGE_1, BigBrotherStateMachine::State::STAGE_2 },
    { BigBrotherStateMachine::State::STAGE_2, BigBrotherStateMachine::State::STAGE_3 },
    { BigBrotherStateMachine::State::STAGE_3, BigBrotherStateMachine::State::STAGE_1 },
};

BigBrotherStateMachine::BigBrotherStateMachine()
{
    // Start at init
    current_state = State::INIT;
    state_callback_map = {};
    is_stage2_done = false;
}

void BigBrotherStateMachine::register_callback(State stage, ButtonPressCallback callback)
{
    state_callback_map.insert(std::make_pair(stage, callback));
}
 
void BigBrotherStateMachine::stage2_complete()
{
    // Signal that the stage is complete 
    is_stage2_done = true;
}

void BigBrotherStateMachine::button_pressed()
{
    // If a callback has been registered for the current state, call it 
    if (state_callback_map.find(current_state) != state_callback_map.end())
    {
        state_callback_map[current_state](current_state);
    }

    // Unless we're in State 2, we transition immediately to the next state 
    if (current_state == State::STAGE_2)
    {
        if (is_stage2_done)
        {
            // Only transition if Stage 2 is done! 
            current_state = state_transition_map[current_state];
            is_stage2_done = false;
        }
    } 
    else 
    {
        current_state = state_transition_map[current_state];
    }
}