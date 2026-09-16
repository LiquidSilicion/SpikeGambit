`timescale 1ns/1ps

module tb_lif_neuron;
    
    // Parameters
    localparam DATA_WIDTH = 16;
    localparam LEAK_SHIFT = 3;
    localparam CLK_PERIOD = 10; // 10ns, 100MHz clock
    
    // Signals
    reg clk;
    reg rst_n;
    reg spike_in;
    reg signed [DATA_WIDTH-1:0] weight;
    reg signed [DATA_WIDTH-1:0] threshold;
    wire spike_out;
    wire signed [DATA_WIDTH-1:0] membrane_potential;
    
    // Instantiate DUT (Device Under Test)
    lif_neuron #(
        .DATA_WIDTH(DATA_WIDTH),
        .LEAK_SHIFT(LEAK_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .spike_in(spike_in),
        .weight(weight),
        .threshold(threshold),
        .spike_out(spike_out),
        .membrane_potential(membrane_potential)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize
        rst_n = 0;
        spike_in = 0;
        weight = 16'sh0100;      // 1.0 in Q8.8 format
        threshold = 16'sh0080;   // 0.5 in Q8.8 format
        
        // Reset
        #100;
        rst_n = 1;
        #50;
        
        // Test 1: Single spike input
        $display("Test 1: Single spike input");
        spike_in = 1;
        #20;
        spike_in = 0;
        #100;
        $display("  Membrane potential after spike: %d", membrane_potential);
        
        // Test 2: Multiple spikes (should accumulate)
        $display("\nTest 2: Multiple spike inputs");
        repeat(5) begin
            spike_in = 1;
            #20;
            spike_in = 0;
            #20;
        end
        #100;
        $display("  Membrane potential after 5 spikes: %d", membrane_potential);
        
        // Test 3: Spike firing (membrane should exceed threshold)
        $display("\nTest 3: Spike firing");
        repeat(10) begin
            spike_in = 1;
            #20;
            spike_in = 0;
            #20;
        end
        #100;
        $display("  Spike output observed: %b", spike_out);
        $display("  Membrane potential after firing: %d", membrane_potential);
        
        // Test 4: Leak behavior (no input, membrane should decay)
        $display("\nTest 4: Leak behavior");
        spike_in = 0;
        #200;
        $display("  Membrane potential after leak: %d", membrane_potential);
        
        $display("\n✅ All tests completed");
        $finish;
    end
    
    // Monitor outputs
    initial begin
        $monitor("Time=%0t | rst_n=%b | spike_in=%b | V_mem=%d | spike_out=%b",
                 $time, rst_n, spike_in, membrane_potential, spike_out);
    end
    
    // Waveform dump
    initial begin
        $dumpfile("lif_neuron.vcd");
        $dumpvars(0, tb_lif_neuron);
    end

endmodule