import logging, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)

class secondary_structure_map:
    def __init__(self):
        self.ss_index = []
        self.ss_type_assignments = []
        self.ss_element_start = []
        self.ss_element_end = []
        self.ss = []

    def assign_ss(self, index_num, ss_type, start, end):
        self.ss_index.append(index_num)
        self.ss_type_assignments.append(ss_type)
        self.ss_element_start.append(start)
        self.ss_element_end.append(end)

    def set_ss(self, ss):
        self.ss = ss

    def map_body_ss(self, body, min_helix_length=4, min_sheet_length=3, min_loop_length=1):
        ss_params = {"H": min_helix_length, "E": min_sheet_length, "L": min_loop_length}

        self.set_ss(body.ss)

        # first residue assignment
        temp_ss = body.ss[0]
        temp_start = 0
        n = 0
        for i, ss_at_resi in enumerate(body.ss):
            if ss_at_resi == temp_ss:  # still in the same ss element
                temp_end = i
            else:  # left the ss element, record it if it is sufficiently long
                end = temp_end
                start = temp_start
                ss_type = temp_ss
                if end - start >= ss_params[ss_type] - 1:
                    # create an SS element entry
                    self.assign_ss(n, ss_type, start, end)
                    n = n+1
                temp_start = i
                temp_ss = ss_at_resi

def filter_sscount(body1, body2, pos1, pos2, min_helix_length=4, min_sheet_length=3, min_loop_length=1, min_element_resis=1, max_dist=8.0,
                   sstype="EHL", confidence=0, min_ss_count=3, simple=True, strict=False, **kw):

    pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
        body1.bvh_cen,
        body2.bvh_cen,
        pos1 @ body1.pos,
        pos2 @ body2.pos,
        max_dist,
    )
    #map ss to an object
    body1_ss_map, body2_ss_map = secondary_structure_map(), secondary_structure_map()
    body1_ss_map.map_body_ss(body1, min_helix_length, min_sheet_length, min_loop_length)
    body2_ss_map.map_body_ss(body2, min_helix_length, min_sheet_length, min_loop_length)
    bod_len = (body1.asym_body.nres, body2.asym_body.nres)

    sscounts_data = []
    ss_counts = np.zeros(max(len(pos1), len(pos2)))
    for i, (lb, ub) in enumerate(lbub):

        #this loses context information.
        body1_res, body2_res = np.unique(pairs[lb:ub][:,0]), np.unique(pairs[lb:ub][:,1])
        #store residues in an SS element 
        temp_result = {
            "A" : {
               "pdb_file" : body1.pdbfile,
               "E" : 0,
               "H" : 0,
               "L" : 0,
               "total_counts" : 0,
               "resis" : [],
               "paired_resis" : []},
            "B" : {
               "pdb_file" : body2.pdbfile,
               "E" : 0,
               "H" : 0,
               "L" : 0,
               "total_counts" : 0,
               "resis" : [],
               "paired_resis" : []},
            "total_count" : 0}
        resis = np.array([])
        for ss_element in body1_ss_map.ss_index:
            start = body1_ss_map.ss_element_start[ss_element]
            end = body1_ss_map.ss_element_end[ss_element]
            ss_type = body1_ss_map.ss_type_assignments[ss_element]
            list_resis = body1_res[(body1_res >= start + 1) & (body1_res <= end + 1)]
            ss_len = len(list_resis)
            if ss_type in sstype and ss_len >= min_element_resis:
                resis = np.append(resis, list_resis)
                temp_result["A"][ss_type] = temp_result["A"][ss_type] + 1

        # get the paired residues
        # Correct the residue numbers to match the input numbering
        paired_resis = np.unique(pairs[lb:ub, 1][np.isin(pairs[lb:ub, 0], resis)])
        resis = resis - (bod_len[0] * (resis / bod_len[0]).astype(int))
        paired_resis = paired_resis - (bod_len[1] * (paired_resis / bod_len[1]).astype(int))
        temp_result["A"]["total_counts"] = temp_result["A"]["H"] + temp_result["A"]["E"] + temp_result["A"]["L"]
        temp_result["A"]["resis"] = resis
        temp_result["A"]["paired_resis"] = paired_resis

        resis = np.array([])
        for ss_element in body2_ss_map.ss_index:
            start = body2_ss_map.ss_element_start[ss_element]
            end = body2_ss_map.ss_element_end[ss_element]
            ss_type = body2_ss_map.ss_type_assignments[ss_element]
            list_resis = body2_res[(body2_res >= start + 1) & (body2_res <= end + 1)]
            ss_len = len(list_resis)
            if ss_type in sstype and ss_len >= min_element_resis:
                resis= np.append(resis, list_resis)
                temp_result["B"][ss_type] = temp_result["B"][ss_type] + 1

        # get the paired residues
        # Correct the residue numbers to match the input numbering
        paired_resis = np.unique(pairs[lb:ub, 0][np.isin(pairs[lb:ub, 1], resis)])
        resis = resis - (bod_len[1] * (resis / bod_len[1]).astype(int))
        paired_resis = paired_resis - (bod_len[0] * (paired_resis / bod_len[0]).astype(int))
        temp_result["B"]["total_counts"] = temp_result["B"]["H"] + temp_result["B"]["E"] + temp_result["B"]["L"]
        temp_result["B"]["resis"] = resis
        temp_result["B"]["paired_resis"] = paired_resis
        temp_result["total_count"] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]

        if strict:
            #Require that, for a residue to count in an SS element, it has to be paired with a residue that is also in an SS element.
            #reset counts
            temp_result["A"]["E"] = 0
            temp_result["A"]["H"] = 0
            temp_result["A"]["L"] = 0
            temp_result["B"]["E"] = 0
            temp_result["B"]["H"] = 0
            temp_result["B"]["L"] = 0
            temp_result["A"]["total_counts"] = 0
            temp_result["B"]["total_counts"] = 0
            temp_result["total_count"] = 0

            #check that the paired residues are also in an SS element.
            resis = np.array([])
            for ss_element in body1_ss_map.ss_index:
                start = body1_ss_map.ss_element_start[ss_element]
                end = body1_ss_map.ss_element_end[ss_element]
                ss_type = body1_ss_map.ss_type_assignments[ss_element]
                body1_resis = temp_result["B"]["paired_resis"]
                list_resis = body1_resis[(body1_resis >= start + 1) & (body1_resis <= end + 1)]
                ss_len = len(list_resis)
                if ss_type in sstype and ss_len >= min_element_resis:
                    resis = np.append(resis, list_resis)
                    temp_result["A"][ss_type] = temp_result["A"][ss_type] + 1
            temp_result["A"]["total_counts"] = temp_result["A"]["H"] + temp_result["A"]["E"] + temp_result["A"]["L"]
            temp_result["A"]["resis"] = resis

            resis = np.array([])
            for ss_element in body2_ss_map.ss_index:
                start = body2_ss_map.ss_element_start[ss_element]
                end = body2_ss_map.ss_element_end[ss_element]
                ss_type = body2_ss_map.ss_type_assignments[ss_element]
                body2_resis = temp_result["A"]["paired_resis"]
                list_resis = body2_resis[(body2_resis >= start + 1) & (body2_resis <= end + 1)]
                ss_len = len(list_resis)
                if ss_type in sstype and ss_len >= min_element_resis:
                    resis = np.append(resis, list_resis)
                    temp_result["B"][ss_type] = temp_result["B"][ss_type] + 1
            temp_result["B"]["total_counts"] = temp_result["B"]["H"] + temp_result["B"]["E"] + temp_result["B"]["L"]
            temp_result["B"]["resis"] = resis
            temp_result["B"]["paired_resis"] = temp_result["A"]["resis"]
            temp_result["A"]["paired_resis"] = temp_result["B"]["resis"]
            temp_result["total_count"] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            ss_counts[i] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            sscounts_data.append(temp_result)
        else:
            ss_counts[i] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            sscounts_data.append(temp_result)

    if confidence==1:
        return ss_counts >= min_ss_count
    elif simple:
        return ss_counts
    else:
        return sscounts_data

