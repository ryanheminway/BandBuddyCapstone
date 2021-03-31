#include "big_brother_state_machine.h" 

#include <stdexcept>
#include <iostream>


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

bool BigBrotherStateMachine::button_pressed()
{
    bool awaitButtonForNextState;


   if (current_state == State::INIT) 
   {
       while(current_state != State::STAGE_3)
       {
            // If a callback has been registered for the current state, call it 
            if (state_callback_map.find(current_state) != state_callback_map.end())
            {
            awaitButtonForNextState = state_callback_map[current_state](current_state);
            }
            current_state = state_transition_map[current_state];
       }
       return awaitButtonForNextState;
   } 
   else if (current_state == State::STAGE_3)
   {
     do
       {
            // If a callback has been registered for the current state, call it 
            if (state_callback_map.find(current_state) != state_callback_map.end())
            {
            awaitButtonForNextState = state_callback_map[current_state](current_state);
            }
            current_state = state_transition_map[current_state];
       } while (current_state != State::STAGE_3);
       
       
       return awaitButtonForNextState;
   }
       

    return awaitButtonForNextState;
}