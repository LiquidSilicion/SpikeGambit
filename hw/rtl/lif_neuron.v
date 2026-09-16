// ============================================================================
// Module: lif_neuron
// Description: Hardware-efficient Leaky Integrate-and-Fire neuron
//              Fixed-point arithmetic: Q8.8 format (8 integer, 8 fractional bits)
// ============================================================================

module lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter LEAK_SHIFT = 3  // Leak factor = 1 - 2^(-LEAK_SHIFT) = 0.875
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    spike_in,       // Input spike (1 bit)
    input  wire signed [DATA_WIDTH-1:0] weight,    // Synaptic weight (Q8.8)
    input  wire signed [DATA_WIDTH-1:0] threshold, // Firing threshold (Q8.8)
    output reg                     spike_out,      // Output spike (1 bit)
    output wire signed [DATA_WIDTH-1:0] membrane_potential  // Debug output
);

    // Membrane potential register (Q8.8 format)
    reg signed [DATA_WIDTH-1:0] V_mem;
    
    // Leak operation: V_leaked = V_mem - (V_mem >> LEAK_SHIFT)
    // This implements V_mem * (1 - 2^(-LEAK_SHIFT))
    wire signed [DATA_WIDTH-1:0] V_leaked;
    
    // Input current: I = spike_in ? weight : 0
    wire signed [DATA_WIDTH-1:0] I_in;
    
    // Leak calculation (arithmetic right shift preserves sign)
    assign V_leaked = V_mem - (V_mem >>> LEAK_SHIFT);
    
    // Input current
    assign I_in = spike_in ? weight : 16'sd0;
    
    // Membrane potential update: V[t] = V_leaked + I_in
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            V_mem <= 16'sd0;
        end else if (spike_out) begin
            // Reset membrane potential after spike (hard reset)
            V_mem <= 16'sd0;
        end else begin
            V_mem <= V_leaked + I_in;
        end
    end
    
    // Spike generation: fire if V_mem >= threshold
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_out <= 1'b0;
        end else begin
            spike_out <= (V_mem >= threshold) ? 1'b1 : 1'b0;
        end
    end
    
    // Debug output
    assign membrane_potential = V_mem;

endmodule