// ============================================================================
// Module: conv_engine
// Description: Spike-driven 3x3 convolution engine
//              Only performs MAC operations when input spike is present
// ============================================================================

module conv_engine #(
    parameter IN_CHANNELS = 12,
    parameter OUT_CHANNELS = 32,
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH = 16,
    parameter BOARD_SIZE = 8
)(
    input  wire clk,
    input  wire rst_n,
    
    // Input spike interface
    input  wire                    spike_in,
    input  wire [3:0]              in_channel,     // 0 to IN_CHANNELS-1
    input  wire [2:0]              x_coord,        // 0 to 7
    input  wire [2:0]              y_coord,        // 0 to 7
    
    // Weight memory interface
    input  wire signed [DATA_WIDTH-1:0] weight_data,
    
    // Output spike interface
    output reg                     spike_out,
    output reg [4:0]               out_channel,    // 0 to OUT_CHANNELS-1
    output reg [2:0]               out_x,
    output reg [2:0]               out_y
);

    // State machine states
    localparam IDLE = 3'd0;
    localparam LOAD_WEIGHTS = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam UPDATE_NEURONS = 3'd3;
    localparam CHECK_SPIKES = 3'd4;
    
    reg [2:0] state;
    
    // Counters for convolution window
    reg [1:0] kx, ky;  // Kernel offsets (0, 1, 2)
    reg [4:0] oc;       // Output channel counter
    
    // Accumulator for current convolution window
    reg signed [DATA_WIDTH-1:0] accumulator;
    
    // Membrane potentials for output neurons (simplified - in real design, use BRAM)
    // For now, we'll use a small subset for demonstration
    reg signed [DATA_WIDTH-1:0] membrane [0:OUT_CHANNELS-1];
    
    // Threshold (fixed for now, can be parameterized)
    localparam signed [DATA_WIDTH-1:0] THRESHOLD = 16'sh0080;  // 0.5 in Q8.8
    
    // Initialize membrane potentials
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                membrane[i] <= 16'sd0;
            end
        end
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            spike_out <= 1'b0;
            accumulator <= 16'sd0;
            kx <= 2'd0;
            ky <= 2'd0;
            oc <= 5'd0;
            out_channel <= 5'd0;
            out_x <= 3'd0;
            out_y <= 3'd0;
        end else begin
            case (state)
                IDLE: begin
                    spike_out <= 1'b0;
                    if (spike_in) begin
                        state <= LOAD_WEIGHTS;
                        kx <= 2'd0;
                        ky <= 2'd0;
                        oc <= 5'd0;
                        accumulator <= 16'sd0;
                    end
                end
                
                LOAD_WEIGHTS: begin
                    // In real design, calculate weight address and fetch from BRAM
                    // For now, weight_data is provided externally
                    state <= COMPUTE;
                end
                
                COMPUTE: begin
                    // Accumulate: acc += spike_in * weight
                    if (spike_in) begin
                        accumulator <= accumulator + weight_data;
                    end
                    
                    // Move to next kernel position
                    if (kx < 2'd2) begin
                        kx <= kx + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else if (ky < 2'd2) begin
                        kx <= 2'd0;
                        ky <= ky + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else if (oc < OUT_CHANNELS - 1) begin
                        // Update neuron membrane for current output channel
                        membrane[oc] <= membrane[oc] + 
                                       (accumulator - (accumulator >>> 3)); // Leak
                        kx <= 2'd0;
                        ky <= 2'd0;
                        oc <= oc + 5'd1;
                        accumulator <= 16'sd0;
                        state <= LOAD_WEIGHTS;
                    end else begin
                        // Last output channel
                        membrane[oc] <= membrane[oc] + 
                                       (accumulator - (accumulator >>> 3));
                        state <= CHECK_SPIKES;
                    end
                end
                
                CHECK_SPIKES: begin
                    // Check if any neuron fired
                    spike_out <= 1'b0;
                    for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                        if (membrane[i] >= THRESHOLD) begin
                            spike_out <= 1'b1;
                            out_channel <= i[4:0];
                            out_x <= x_coord;
                            out_y <= y_coord;
                            membrane[i] <= 16'sd0;  // Reset after spike
                        end
                    end
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule