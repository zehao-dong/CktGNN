import numpy as np
import os
import copy
import csv


def cktgraph_to_fom(cktgrpah_path):

    subg_node = {
        
        0: ['In'],
        1: ['Out'],
        2: ['R'],
        3: ['C'],
        4: ['R', 'C'],
        5: ['R', 'C'],
        6: ['+gm+'],
        7: ['-gm+'],
        8: ['+gm-'],
        9: ['-gm-'],
        10: ['C', '+gm+'],
        11: ['C', '-gm+'],
        12: ['C', '+gm-'],
        13: ['C', '-gm-'],
        14: ['R', '+gm+'],
        15: ['R', '-gm+'],
        16: ['R', '+gm-'],
        17: ['R', '-gm-'],
        18: ['C', 'R', '+gm+'],
        19: ['C', 'R', '-gm+'],
        20: ['C', 'R', '+gm-'],
        21: ['C', 'R', '-gm-'],
        22: ['C', 'R', '+gm+'],
        23: ['C', 'R', '-gm+'],
        24: ['C', 'R', '+gm-'],
        25: ['C', 'R', '-gm-']
    }


    polarity_node = {
        0: None,
        1: None,
        2: ['1'],
        3: ['1'],
        4: ['1', '1'],
        5: ['1', '1'],
        6: ['1'],
        7: ['-1'],
        8: ['1'],
        9: ['-1'],
        10: ['1', '1'],
        11: ['1', '-1'],
        12: ['1', '1'],
        13: ['1', '-1'],
        14: ['1', '1'],
        15: ['1', '-1'],
        16: ['1', '1'],
        17: ['1', '-1'],
        18: ['1', '1', '1'],
        19: ['1', '1', '-1'],
        20: ['1', '1', '1'],
        21: ['1', '1', '-1'],
        22: ['1', '1', '1'],
        23: ['1', '1', '-1'],
        24: ['1', '1', '1'],
        25: ['1', '1', '-1']
    }


    basis_node = {
        0: None,
        1: None,
        2: ['0.1'],
        3: ['10'],
        4: ['0.1', '10'],
        5: ['0.1', '10'],
        6: ['0.1'],
        7: ['0.1'],
        8: ['0.1'],
        9: ['0.1'],
        10: ['10', '0.1'],
        11: ['10', '0.1'],
        12: ['10', '0.1'],
        13: ['10', '0.1'],
        14: ['0.1', '0.1'],
        15: ['0.1', '0.1'],
        16: ['0.1', '0.1'],
        17: ['0.1', '0.1'],
        18: ['10', '0.1', '0.1'],
        19: ['10', '0.1', '0.1'],
        20: ['10', '0.1', '0.1'],
        21: ['10', '0.1', '0.1'],
        22: ['10', '0.1', '0.1'],
        23: ['10', '0.1', '0.1'],
        24: ['10', '0.1', '0.1'],
        25: ['10', '0.1', '0.1']
    }

    unit_node = {
        0: None,
        1: None,
        2: ['M'],
        3: ['f'],
        4: ['M', 'f'],
        5: ['M', 'f'],
        6: ['m'],
        7: ['m'],
        8: ['m'],
        9: ['m'],
        10: ['f', 'm'],
        11: ['f', 'm'],
        12: ['f', 'm'],
        13: ['f', 'm'],
        14: ['M', 'm'],
        15: ['M', 'm'],
        16: ['M', 'm'],
        17: ['M', 'm'],
        18: ['f', 'M', 'm'],
        19: ['f', 'M', 'm'],
        20: ['f', 'M', 'm'],
        21: ['f', 'M', 'm'],
        22: ['f', 'M', 'm'],
        23: ['f', 'M', 'm'],
        24: ['f', 'M', 'm'],
        25: ['f', 'M', 'm']
    }




    direction_node = {
        0: None,
        1: None,
        2: ['+'],
        3: ['+'],
        4: ['+'],
        5: ['+'],
        6: ['+'],
        7: ['+'],
        8: ['-'],
        9: ['-'],
        10: ['+'],
        11: ['+'],
        12: ['-'],
        13: ['-'],
        14: ['+'],
        15: ['+'],
        16: ['-'],
        17: ['-'],
        18: ['+'],
        19: ['+'],
        20: ['-'],
        21: ['-'],
        22: ['+'],
        23: ['+'],
        24: ['-'],
        25: ['-']
    }




    ##### subnetlists ara corrected #####

    subnetlist_node = {0: None,
        1: None,
        2: ["subckt single_r_2 IN OUT \n", "    R0 (IN OUT) resistor r=? \n", "ends single_r_2 \n"],
        3: ["subckt single_c_3 IN OUT \n", "    C0 (IN OUT) capacitor c=? \n", "ends single_c_3 \n"],
        4: ["subckt r_c_s_4 IN OUT \n", "    R0 (IN net1) resistor r=? \n", 
        "  C0 (net1 OUT) capacitor c=? \n", "ends r_c_s_4 \n"],
        5: ["subckt r_c_p_5 IN OUT \n", "    R0 (IN OUT) resistor r=? \n", 
        "  C0 (IN OUT) capacitor c=? \n", "ends r_c_p_5 \n"],
        6: ["subckt tc_stage_6 GND OUT IN \n", "    G0 (OUT GND IN GND) vccs gm=? \n", 
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n", 
        "ends tc_stage_6 \n"],
        7: ["subckt tc_stage_7 GND OUT IN \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",
        "ends tc_stage_7 \n"],
        8: ["subckt tc_stage_8 GND OUT IN \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",
        "ends tc_stage_8 \n"],
        9: ["subckt tc_stage_9 GND OUT IN \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",
        "ends tc_stage_9 \n"],
        10: ["subckt ts_c_p_10 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n", 
        "    C0 (OUT GND) capacitor c=50.0f \n",     
        "ends ts_c_p_10 \n"],
        11: ["subckt ts_c_p_11 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n",
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n", 
        "    C0 (OUT GND) capacitor c=50.0f \n",  
        "ends ts_c_p_11 \n"],
        12: ["subckt ts_c_p_12 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n",
        "    C0 (OUT GND) capacitor c=50.0f \n",
        "ends ts_c_p_12 \n"],
        13: ["subckt ts_c_p_13 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n", 
        "    C0 (OUT GND) capacitor c=50.0f \n",     
        "ends ts_c_p_13 \n"],
        14: ["subckt ts_r_p_14 GND IN OUT \n", "    R1 (IN OUT) resistor r=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n",
        "    C0 (OUT GND) capacitor c=50.0f \n",
        "ends ts_r_p_14 \n"],
        15: ["subckt ts_r_p_15 GND IN OUT \n", "    R1 (IN OUT) resistor r=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n", 
        "    C0 (OUT GND) capacitor c=50.0f \n",     
        "ends ts_r_p_15 \n"],
        16: ["subckt ts_r_p_16 GND IN OUT \n", "    R1 (IN OUT) resistor r=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n",
        "    C0 (OUT GND) capacitor c=50.0f \n",     
        "ends ts_r_p_16 \n"],
        17: ["subckt ts_r_p_17 GND IN OUT \n", "    R1 (IN OUT) resistor r=? \n", 
        "    G0 (OUT GND IN GND) vccs gm=? \n", "    R0 (OUT GND) resistor r=1M \n", 
        "    C0 (OUT GND) capacitor c=50.0f \n",     
        "ends ts_r_p_17 \n"],
        18: ["subckt ts_r_c_p_18 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n",
        "    R1 (IN OUT) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n", 
        "ends ts_r_c_p_18 \n"],
        19: ["subckt ts_r_c_p_19 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n",
        "    R1 (IN OUT) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n", 
        "ends ts_r_c_p_19 \n"],
        20: ["subckt ts_r_c_p_20 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n",
        "    R1 (IN OUT) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n", 
        "ends ts_r_c_p_20 \n"],
        21: ["subckt ts_r_c_p_21 GND IN OUT \n", "    C1 (IN OUT) capacitor c=? \n",
        "    R1 (IN OUT) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n", 
        "ends ts_r_c_p_21 \n"],
        22: ["subckt ts_r_c_s_22 GND IN OUT \n", "    C1 (net1 OUT) capacitor c=? \n",
        "    R1 (IN net1) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",  
        "ends ts_r_c_s_22 \n"],
        23: ["subckt ts_r_c_s_23 GND IN OUT \n", "  C1 (net1 OUT) capacitor c=? \n",
        "  R1 (IN net1) resistor r=? \n", "  G0 (OUT GND IN GND) vccs gm=? \n",
        "  R0 (OUT GND) resistor r=1M \n", "  C0 (OUT GND) capacitor c=50.0f \n",  
        "ends ts_r_c_s_23 \n"],
        24: ["subckt ts_r_c_s_24 GND IN OUT \n", "    C1 (net1 OUT) capacitor c=? \n",
        "    R1 (IN net1) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",  
        "ends ts_r_c_s_24 \n"],
        25: ["subckt ts_r_c_s_25 GND IN OUT \n", "    C1 (net1 OUT) capacitor c=? \n",
        "    R1 (IN net1) resistor r=? \n", "    G0 (OUT GND IN GND) vccs gm=? \n",
        "    R0 (OUT GND) resistor r=1M \n", "    C0 (OUT GND) capacitor c=50.0f \n",  
        "ends ts_r_c_s_25 \n"]
    }

    ##### subnetlists ara corrected #####


    ##### get trained parameter from each row #####

    def get_para(intermediate_row):

        j = 0

        index_list = []

        while j < len(intermediate_row):

            if "-1" == intermediate_row[j]:

                index_list.append(j)

            j += 1

        return index_list

    ##############################################



    with open(cktgrpah_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]



    number_row_text = len(lines)

    row_start = 0
    node_start = 0
    current_circuit_start_point = 0



    while row_start < number_row_text:

        ##### each element in a list is a string, split is used to sperate this string into a list ######
        intermediate_row = lines[row_start].split() 
        #################################################################################################




        if row_start == current_circuit_start_point:



            number_of_sub_graph = int(intermediate_row[0])
            number_of_nodes_circuit = int(intermediate_row[1])
            op_stage_type = int(intermediate_row[2])

            nodetype_name_dic = {}
            subcircuit_name_dic = {}
            inputnode_name_dic = {}
            outputnode_name_dic = {}
            nodeposition_name_dic = {}
            subcircuit_node_name_dic = {}
            subcircuit_node_name_new_dic = {}
            circuit_netlist_dic = {}
            unique_subcircuit_num_dic = {}
            subcircuit_name_new_dic = {}



            ##### detect the type of op-amp #####
            if op_stage_type == 3:

                inputnode_name_dic.update({1: 'net1'})
                inputnode_name_dic.update({2: 'net2'})
                inputnode_name_dic.update({3: 'net3'})
                inputnode_name_dic.update({4: 'net4'})

                outputnode_name_dic.update({0: 'net2'})
                outputnode_name_dic.update({1: 'net1'})
                outputnode_name_dic.update({2: 'net3'})
                outputnode_name_dic.update({3: 'net4'})
                outputnode_name_dic.update({4: 'net1'})

            if op_stage_type == 2:

                inputnode_name_dic.update({1: 'net1'})
                inputnode_name_dic.update({2: 'net2'})
                inputnode_name_dic.update({3: 'net3'})

                outputnode_name_dic.update({0: 'net2'})
                outputnode_name_dic.update({1: 'net1'})
                outputnode_name_dic.update({2: 'net3'})
                outputnode_name_dic.update({3: 'net1'})
            ######################################

            ##### dic initilization #####
            while node_start < number_of_sub_graph:

                nodetype_name_dic.update({node_start: None})
                subcircuit_name_dic.update({node_start: None}) 
                nodeposition_name_dic.update({node_start: None})
                subcircuit_node_name_dic.update({node_start: None})
                subcircuit_node_name_new_dic.update({node_start: None})
                circuit_netlist_dic.update({node_start: None})
                subcircuit_name_new_dic.update({node_start: None})

                node_start += 1
            #############################

            #row_start += 1


        if (row_start > current_circuit_start_point) and (row_start < (current_circuit_start_point + number_of_sub_graph + 1)):

            connection_start = 0

            ##### get the node type
            converted_num_index_0 = int(intermediate_row[0])

            ##### get the node index
            converted_num_index_1 = int(intermediate_row[1])

            ##### get the node position
            converted_num_index_2 = int(intermediate_row[2])

            ##### get the node connection number
            #print(intermediate_row)
            converted_num_index_3 = int(intermediate_row[3])

            ##### the position node of the sub_circuit 
            nodeposition_name_dic.update({converted_num_index_1: converted_num_index_2})

            ##### the name of the sub_circuit 
            if converted_num_index_0 != 0 and converted_num_index_0 != 1:

                nodetype_name_dic.update({converted_num_index_1: converted_num_index_0})

                ##### get the netlist of each sub_graph #####
                node_netlist = copy.deepcopy(subnetlist_node[converted_num_index_0])
                #############################################

                subcircuit_name_dic.update({converted_num_index_1: node_netlist[0].split()[1]})
                subcircuit_node_name_dic.update({converted_num_index_1: node_netlist[0].split()[2:]})
                subcircuit_node_name_new_dic.update({converted_num_index_1: node_netlist[0].split()[2:]})

                #print(node_netlist[0].split()[1])
                ##### check if there are the same subgraphs used in the design #####
                if node_netlist[0].split()[1] in unique_subcircuit_num_dic:

                    #print(node_netlist[0].split()[1])

                    unique_subcircuit_num_dic.update({node_netlist[0].split()[1]: unique_subcircuit_num_dic[node_netlist[0].split()[1]] + 1})

                else: 

                    unique_subcircuit_num_dic.update({node_netlist[0].split()[1]: 0})

                #print(unique_subcircuit_num_dic)
                #print(subnetlist_node[converted_num_index_0])
                #print(node_netlist[0].split()[1])
                subcircuit_name_new_dic.update({converted_num_index_1: node_netlist[0].split()[1] + '_' + str(unique_subcircuit_num_dic[node_netlist[0].split()[1]])})
                #print(subcircuit_name_new_dic)
                ####################################################################



                ##### update the subcircuit name in the generated circuit netlist #####
                netlist_name = node_netlist[0].split()[1]

                ##### issue happens here // fixed by using deepcopy #####
                inter_name_head = node_netlist[0].replace(netlist_name, subcircuit_name_new_dic[converted_num_index_1])
                inter_name_tail = node_netlist[-1].replace(netlist_name, subcircuit_name_new_dic[converted_num_index_1])
                ##############################

                #print(inter_name_head)
                node_netlist[0] = inter_name_head
                #print(subnetlist_node[converted_num_index_0])


                node_netlist[-1] = inter_name_tail
                #print(subnetlist_node[converted_num_index_0])

                #######################################################################


                ##### get index of -1 where trained parameters #####
                index_list = get_para(intermediate_row)
                ####################################################

                s_var_numb_s_point = 0

                intermedaite_updated_variable = []
                subnetlist_variable_number = int(index_list[1]) - (int(index_list[0]) + 1)


                ##### get trained parameters #####
                while s_var_numb_s_point < subnetlist_variable_number:

                    ##### get basis #####                
                    basis = float(basis_node[converted_num_index_0][s_var_numb_s_point].split()[0])
                    # print(basis)
                    #####################

                    ##### consider the polarity of the trained parameters #####
                    intermedaite_updated_value_polarity = basis * float(intermediate_row[int(index_list[0]) + 1 \
                    + s_var_numb_s_point]) * int(polarity_node[converted_num_index_0][s_var_numb_s_point])
                    ###########################################################


                    part_net_list = node_netlist[1 + s_var_numb_s_point].split()               
                    if '=' in part_net_list[-1]:
                        # print('true')
                        inter_v = part_net_list[-1].replace('?', str(intermedaite_updated_value_polarity) + unit_node[converted_num_index_0][s_var_numb_s_point])

                    ##### update the netlist of the subgraph with trained parameter #####    
                    part_net_list[-1] = inter_v
                    node_netlist[1 + s_var_numb_s_point] =' ' + ' ' +  ' ' + ' ' + ' '.join(part_net_list) + '\n'
                    circuit_netlist_dic.update({converted_num_index_1: node_netlist})
                    # print('circuit_netlist_dic')
                    # print(circuit_netlist_dic)
                    # print('\n')
                    #####################################################################  

                    s_var_numb_s_point += 1

            ##### get outputnode #####
            if converted_num_index_3 != 0:

                while connection_start < converted_num_index_3:

                    if int(intermediate_row[3 + 1 + connection_start]) > (op_stage_type + 1):

                        outputnode_name_dic.update({int(intermediate_row[3 + 1 + connection_start]): outputnode_name_dic[converted_num_index_1]})
            ##########################

            ##### get inputnode ######
                    if converted_num_index_1 > (op_stage_type + 1):
                        inputnode_name_dic.update({converted_num_index_1: outputnode_name_dic[int(intermediate_row[3 + 1 + connection_start])]})
            ##########################

                    connection_start += 1



        ##### set starting point for the next circuit #####
        if row_start == (current_circuit_start_point + 1 + number_of_sub_graph):

            ##### here, row_start indicates the end row index of current circuit ##### 
            row_start = row_start + number_of_nodes_circuit - 1
            ##########################################################################
            next_circuit_stat_point = row_start + 1
            current_circuit_start_point = next_circuit_stat_point

            #print('row_start, current_circuit_end_point')
            #print(row_start, current_circuit_start_point)



            for key in nodetype_name_dic:

                i = 0

                if key != 0 and key != 1:

                    while i < len(subcircuit_node_name_dic[key]):

                        if subcircuit_node_name_dic[key][i] == 'IN':

                            ##### exchange the input/output relationship of the subgraph according to itd feedforward/feedback type #####
                            if direction_node[nodetype_name_dic[key]] == ['-']:

                                #print('feedback instance')
                                #print(nodetype_name_dic[key])

                                subcircuit_node_name_new_dic[key][i] = outputnode_name_dic[key]

                            else:

                                subcircuit_node_name_new_dic[key][i] = inputnode_name_dic[key]
                            #############################################################################################################


                        if subcircuit_node_name_dic[key][i] == 'OUT':

                            subcircuit_node_name_new_dic[key][i] = outputnode_name_dic[key]

                            ##### exchange the input/output relationship of the subgraph according to itd feedforward/feedback type #####
                            if direction_node[nodetype_name_dic[key]] == ['-']:

                                subcircuit_node_name_new_dic[key][i] = inputnode_name_dic[key]

                            else:

                                subcircuit_node_name_new_dic[key][i] = outputnode_name_dic[key]
                            #############################################################################################################

                        if subcircuit_node_name_dic[key][i] == 'GND':

                            subcircuit_node_name_new_dic[key][i] = '0'

                        i += 1




            ##### starting converting netlist #####

            print('\n')

            file1 = open('/home/research/WG-caow/simulation/test/spectre/schematic/netlist/netlist', 'w')
            #print('\n')
            file1.writelines('\n') 
            file1.writelines('// Library name: GNN_Circuit \n') 
            file1.writelines('// Cell name: behavioral_op_amp \n') 
            file1.writelines('// View name: schematic \n') 


            ##### write the netlist of each instance #####
            for key, value in circuit_netlist_dic.items():

                if value != None:

                    file1.writelines(value)

            file1.writelines('\n')

            #############################################

            file1.writelines('// Library name: GNN_Circuit \n') 
            file1.writelines('// Cell name: behavioral_op_amp_test \n') 
            file1.writelines('// View name: schematic \n') 

            #print(subcircuit_node_name_new_dic)

            ##### write the connect of each instance #####
            for key, value in subcircuit_node_name_new_dic.items():

                if key != 0 and key != 1:

                    L1 = ' '.join(value) 
                    if 'net1' in L1:

                        L1 = L1.replace('net1', 'OUT') 

                    L2 = 'I' + str(key - 2) + ' ' +  '('+ L1 +')' + ' ' + subcircuit_name_new_dic[key] + str('\n')
                    file1.writelines(L2)

            ##############################################
            file1.writelines('V0 (net2 0) vsource dc=0 mag=1m type=sine ampl=1m freq=1K \n')
            file1.writelines('\n')
            file1.close()
            ##############################################

            ##############################################


            ##### call cadence for simulation #####      
            os.system('ocean -nograph -replay test.ocn -log opamp.log')


            inter_result = ' '

            with open('results.csv', 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    inter_result += ' '.join(row)

            metric = ' '.join(inter_result.split())

            if 'G' in metric:

                metric = metric.replace('G', 'e9')

            if 'M' in metric:

                metric = metric.replace('M', 'e6')

            if 'm' in metric:

                metric = metric.replace('m', 'e-3')

            if 'u' in metric:

                metric = metric.replace('u', 'e-6')

            if 'n' in metric:

                metric = metric.replace('n', 'e-9')

            if 'p' in metric:

                metric = metric.replace('p', 'e-12')

            if 'f' in metric:

                metric = metric.replace('f', 'e-15')

            if 'K' in metric:

                metric = metric.replace('K', 'e3')

            metric = metric.split(' ')
            #print(metric)

            if len(metric) == 3:

                gain = float(metric[0])
                pm = float(metric[1])
                ugw = float(metric[2])
                fom = 1.2 * np.abs(gain) / 100 + 1.6 * pm / (-90) + 10 * np.abs(ugw) / 1e9

                #print(len(metric))
                #print(metric)
                #print(gain, pm, ugw)
                #print(fom)

            else:
            
                fom = float('-inf')

            metric.clear()
            #print(metric)
            os.remove('results.csv')

            #######################################

        row_start += 1

        ###################################################

    return fom




