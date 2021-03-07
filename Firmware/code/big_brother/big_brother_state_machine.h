#ifndef BIG_BROTHER_STATE_MACHINE_H
#define BIG_BROTHER_STATE_MACHINE_H

#include <unordered_map>

// The state machine that backs Big Brother.
class BigBrotherStateMachine
{
public:

    // The states that this state machine may contain.
    enum class State
    {
        INIT, 
        STAGE_1, 
        STAGE_2, 
        STAGE_3
    };

    // A callback typedef - used to register a button press callback for a State
    using ButtonPressCallback = void(*)(State);

private: 

    // The current state of the state machine.
    State current_state;

    // A map between a State and the callback to invoke if a button is pressed during that State.
    std::unordered_map<State, ButtonPressCallback> state_callback_map;

    // A flag toggled when the caller notifies that Stage 2 has completed.
    bool is_stage2_done;


    // Transition map between States.
    static std::unordered_map<State, State> state_transition_map;

 
public:

    // Initialize the state machine and cycle it to its starting state
    BigBrotherStateMachine(); 

    // For sanity, delete the other crap
    BigBrotherStateMachine(const BigBrotherStateMachine& other) = delete;
    BigBrotherStateMachine(BigBrotherStateMachine&& other) = delete;
    BigBrotherStateMachine& operator=(const BigBrotherStateMachine& other) = delete;
    BigBrotherStateMachine& operator=(BigBrotherStateMachine&& other) noexcept = delete;

    // Register a press callback for a given Stage
    void register_callback(State stage, ButtonPressCallback callback);

    // Notify the state machine that the button was pressed.
    void button_pressed();

    // Notify the state machine that the backbone has completed stage 2
    void stage2_complete();
};

#endif