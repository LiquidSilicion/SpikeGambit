// ============================================================================
// Module: output_classifier
// Description: Accumulates output spikes and determines predicted class
//              Implements rate coding readout
//              Strictly Verilog-2001 compliant (no SystemVerilog features)
// ============================================================================

module output_classifier (
    input  wire clk,
    input  wire rst_n,
    
    // Input spike interface
    input  wire spike_in,
    input  wire [CLASS_ID_WIDTH-1:0] class_id,
    input  wire input_valid,
    
    // Output
    output reg [CLASS_ID_WIDTH-1:0] predicted_class,
    output reg output_valid,
    
    // Control
    input  wire start,
    output reg busy
);

    // Parameters (Verilog-2001 compliant)
    parameter NUM_CLASSES = 2;
    parameter TIME_STEPS = 200;
    parameter DATA_WIDTH = 16;
    parameter CLASS_ID_WIDTH = 1; // 1 bit width for 2 classes (0 or 1)

    // Spike counters for each class (Verilog-2001 memory declaration)
    reg [7:0] spike_count [0:NUM_CLASSES-1];
    
    // Time step counter
    reg [7:0] time_step;
    
    // State machine states (using parameter instead of localparam)
    parameter IDLE = 2'd0;
    parameter COUNTING = 2'd1;
    parameter DECIDE = 2'd2;
    
    reg [1:0] state;
    
    // Loop variables and temporary registers (module level for strict compatibility)
    integer i;
    reg [CLASS_ID_WIDTH-1:0] max_class;
    reg [7:0] max_count;

    // Initialize counters on reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                spike_count[i] <= 8'd0;
            end
        end
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            time_step <= 8'd0;
            output_valid <= 1'b0;
            busy <= 1'b0;
            predicted_class <= {CLASS_ID_WIDTH{1'b0}}; // Zero-initialize to width
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= COUNTING;
                        busy <= 1'b1;
                        time_step <= 8'd0;
                        output_valid <= 1'b0;
                        // Reset counters
                        for (i = 0; i < NUM_CLASSES; i = i + 1) begin
                            spike_count[i] <= 8'd0;
                        end
                    end
                end
                
                COUNTING: begin
                    if (input_valid && spike_in) begin
                        spike_count[class_id] <= spike_count[class_id] + 8'd1;
                    end
                    
                    if (time_step < (TIME_STEPS - 1)) begin
                        time_step <= time_step + 8'd1;
                    end else begin
                        state <= DECIDE;
                    end
                end
                
                DECIDE: begin
                    // Find class with maximum spike count (argmax)
                    max_class = {CLASS_ID_WIDTH{1'b0}};
                    max_count = spike_count[0];
                    
                    for (i = 1; i < NUM_CLASSES; i = i + 1) begin
                        if (spike_count[i] > max_count) begin
                            max_count = spike_count[i];
                            max_class = i[CLASS_ID_WIDTH-1:0]; // Truncate to width
                        end
                    end
                    
                    predicted_class <= max_class;
                    output_valid <= 1'b1;
                    busy <= 1'b0;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule