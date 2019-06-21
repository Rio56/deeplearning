# -*-coding:utf8 -*-
'''
	File Name：     PDBParseBase2.0.2
	Description :   Finish 15 words parser
	Author :        Zhai Yuhao
	date：          2018/1/29
	modification：  2018/2/11

'''

class PDBParserBase():



    """get PDB file informations of what we want.These code will finish 15words
    """
    def __init__(self):
        pass

    # 1
    def get_header_info(self, PDBfile):
        """Get header information.
        The HEADER record uniquely identifies a PDB entry through the idCode field. This record also provides a
        classification for the entry. Finally, it contains the date when the coordinates were deposited to the
        PDB archive.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            HEADER_classification :Classifies the molecule(s).
            HEADER_depDate :Deposition date. This is the date the coordinates  were received at the PDB.
            HEADER_idCode :This identifier is unique within the PDB.
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
        Returns:
                HEADER:One dict which recode the HEADER information.
        Examples:
            {'pdb_id': '220L', 'HEADER_classification': 'YDROLASE', 'HEADER_depDate': '5-JUN-97',
            'HEADER_idCode': '220L'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        HEADER = {}
        for g in range(0, 10):
            line = lines[g]
            ##print(line)
            header = line.split()[0]
            if header == 'HEADER':
                ##print(line)
                pdb_id = self.__parse_PDB_ID_Line(line)
                HEADER_classification = line[10:50].strip()
                HEADER_depDate = line[50:59].strip()
                HEADER_idCode = line[62:66].strip()
        # put key_values for dic
        HEADER['pdb_id'] = pdb_id
        HEADER['HEADER_classification'] = HEADER_classification
        HEADER['HEADER_depDate'] = HEADER_depDate
        HEADER['HEADER_idCode'] = HEADER_idCode
        ##print(HEADER)
        return HEADER

    # 2
    def get_title_info(self, PDBfile):
        """Get title information
            The TITLE record contains a title for the experiment or analysis that is represented in the entry.
         It should identify an entry in the same way that a citation title identifies a publication.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            TITLE_Continuation:Allows concatenation of multiple records.
            TITLE_title:Title of the  experiment.
            TITLE_title_temp:Title of the  experiment,store for temp
        Used functions:
            __load_file:A internal function which can read PDB file .
        Returns:
            TITLE:One dict which recode the TITLE information.
        Examples:
            {'pdb_id': '220L', 'TITLE_Continuation': '2',
            'TITLE_title': 'GENERATING LIGAND BINDING SITES IN T4 LYSOZYME USING DEFICIENCY-CREATING SUBSTITUTIONS'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        #print (lines)
        # define variables
        TITLE = {}
        TITLE_title = ''
        TITLE_Continuation = ''
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'TITLE':
                TITLE_Continuation = line[8:10].strip()  # get how many lines does it have.The number 9 maybe false
                TITLE_title_temp = line[10:81].strip()
                if (TITLE_Continuation):  # if continuation has number put a blak to connect strings
                    TITLE_title = TITLE_title + " " + TITLE_title_temp
                else:  # for the first time do not need a blank
                    TITLE_title = TITLE_title + TITLE_title_temp
        TITLE['pdb_id'] = pdb_id
        TITLE['TITLE_Continuation'] = TITLE_Continuation
        TITLE['TITLE_title'] = TITLE_title
        #print(TITLE)
        return TITLE

    # 3
    def get_compnd_info(self, PDBfile):
        """Get COMPND information
            The COMPND record describes the macromolecular contents of an entry. Some cases where the entry contains a
             standalone drug or inhibitor, the name of the non-polymeric molecule will appear in this record.
            Each macromolecule found in the entry is described by a set of token: value pairs, and is referred to as
            a COMPND record component. Since the concept of a molecule is difficult to specify exactly, staff may
            exercise editorial judgment in consultation with depositors in assigning these names.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                COMPND_Continuation:Allows concatenation of multiple records.
                COMPND _Specification
                COMPND _Specification _temp
                COMPND_MOL_ID:Numbers each component; also used in  SOURCE to associate the information.
                COMPND_MOLECULE:Name of the macromolecule.
                COMPND_CHAIN:Comma-separated list of chain  identifier(s).
                COMPND_FRAGMENT:Specifies a domain or region of the  molecule.
                COMPND_SYNONYM:Comma-separated list of synonyms for  the MOLECULE.
                COMPND_EC:The Enzyme Commission number associated  with the molecule.If there is more than one EC number,  they are presented as a comma-separated list.
                COMPND_ENGINEERED:Indicates that the molecule was  produced using recombinant technology or by purely  chemical synthesis.
                COMPND_MUTATION:Indicates if there is a mutation.
                COMPND_OTHER_DETAILS:Additional comments.

            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                COMPND:One dict which recode the COMPND information.
            Examples:
                {'COMPND_Specification_1':
                {'COMPND_MOL_ID': '1', 'COMPND_MOLECULE': 'T4 LYSOZYME;', 'COMPND_CHAIN': 'A;',
                'COMPND_EC': 'C: 3.2.1.17;', 'COMPND_ENGINEERED': 'YES;', 'COMPND_MUTATION': 'YES;',
                 'COMPND_OTHER_DETAILS': 'BENZENE LIGANDED'},
                 'COMPND_Specification_2':
                 {'COMPND_MOL_ID': '2', 'COMPND_MOLECULE': 'T2 LYSOZYME;', 'COMPND_CHAIN': 'AA;',
                  'COMPND_EC': 'C: 32.23.11.173;', 'COMPND_ENGINEERED': 'YES;', 'COMPND_MUTATION': 'YES;',
                   'COMPND_OTHER_DETAILS': 'BENZENE LIGANDED     ;   123BENZENE LIGANDEDA123'},
                   'pdb_id': '220L', 'COMPND_Continuation': '17', 'COMPND_Specification_num': '2'}

            Raises:
                'FRAGMENT'is not apperence in the test file.So there is no items named"FTAGMENT in
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        COMPND = {}
        COMPND_Continuation = ''
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'COMPND':
                COMPND_Continuation = line[8:10].strip()  # get how many lines does it have.The number 9 maybe false
                #  #get the line number
                # if MOL_ID appeared ,COMPND _Specification id+1
                if 'MOL_ID' == line[10:16].strip() or 'MOL_ID' == line[10:17].strip():
                    # tips: because of strip will let the whiteblank away so it is ok to put[10:17]
                    # if it is first,it is[10:16];other case is[10:17]
                    # it is where to put codes in order to divide items in one mode
                    if ('MOL_ID' == line[10:16].strip()):  # it is mol_id 1
                        COMPND_Specification_temp = 'COMPND_Specification_1'
                        COMPND[COMPND_Specification_temp] = {}
                        COMPND[COMPND_Specification_temp]['COMPND_MOL_ID'] = line[17:19].strip()
                        COMPND_Specification_num = line[17:19].strip()  # if there is only 1 mol.
                        pass
                    elif ('MOL_ID' == line[10:17].strip()):  # it is mol_id next
                        COMPND_Specification_temp = 'COMPND_Specification_' + str(
                            line[18:20].strip())  # put the id_number next the variable
                        COMPND[COMPND_Specification_temp] = {}
                        COMPND[COMPND_Specification_temp]['COMPND_MOL_ID'] = line[18:20].strip()
                        COMPND_Specification_num = line[18:20].strip()
                    pass
                if ('MOLECULE' == line[11:19].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_MOLECULE'] = line[20:80].strip()
                elif ('CHAIN' == line[11:16].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_CHAIN'] = line[17:80].strip()
                    pass
                elif ('FRAGMENT' == line[11:19].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_FRAGMENT'] = line[20:80].strip()
                    pass
                elif ('SYNONYM' == line[11:18].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_SYNONYM'] = line[20:80].strip()
                    pass
                elif ('EC' == line[11:13].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_EC'] = line[12:80].strip()
                    pass
                elif ('ENGINEERED' == line[11:21].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_ENGINEERED'] = line[22:80].strip()
                    pass
                elif ('MUTATION' == line[11:19].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_MUTATION'] = line[20:80].strip()
                    pass
                elif ('OTHER_DETAILS' == line[11:24].strip()):
                    COMPND[COMPND_Specification_temp]['COMPND_OTHER_DETAILS'] = line[25:80].strip()
                    pass
        # #print(COMPND)
        COMPND['pdb_id'] = pdb_id
        COMPND['COMPND_Continuation'] = COMPND_Continuation
        COMPND['COMPND_Specification_num'] = COMPND_Specification_num
        #print(COMPND)
        return COMPND

    # 4
    def get_source_info(self, PDBfile):
        """Get SOURCE information
            The SOURCE record specifies the biological and/or chemical source of each biological molecule in the entry.
            Some cases where the entry contains a standalone drug or inhibitor, the source information of this molecule
            will appear in this record. Sources are described by both the common name and the scientific name, e.g.,
            genus and species. Strain and/or cell-line for immortalized cells are given when they help to uniquely
            identify the biological entity studied.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                SOURCE: return a dic include items source record
                SOURCE_Continuation	:Allows concatenation of multiple records.
                SOURCE_Specification	:
                SOURCE_Specification_temp	:
                SOURCE_Specification_num :recode the numbers of mol_id
                SOURCE_MOL_ID:Numbers each  molecule. Same as appears in COMPND.
                SOURCE_SYNTHETIC:Indicates a  chemically-synthesized source.
                SOURCE_FRAGMENT:A domain or  fragment of the molecule may be  specified.
                SOURCE_ORGANISM_SCIENTIFIC:Scientific name of the  organism.
                SOURCE_ORGANISM_COMMON:Common name of the  organism.
                SOURCE_ORGANISM_TAXID:NCBI Taxonomy ID number  of the organism.
                SOURCE_STRAIN:Identifies the  strain.
                SOURCE_VARIANT:Identifies the  variant.
                SOURCE_CELL_LINE:The specific line of  cells used in the experiment.
                SOURCE_ATCC	:American Type  Culture Collection tissue culture  number.
                SOURCE_ORGAN:Organized group of  tissues that carries on  a specialized function.
                SOURCE_TISSUE:Organized group  of cells with a common     function and  structure.
                SOURCE_CELL:Identifies the  particular cell type.
                SOURCE_ORGANELLE:Organized structure  within a cell.
                SOURCE_SECRETION:Identifies the secretion, such as  saliva, urine, or venom,  from which the molecule was isolated.
                SOURCE_CELLULAR_LOCATION:Identifies the location  inside/outside the cell.
                SOURCE_PLASMID:Identifies the plasmid  containing the gene.
                SOURCE_GENE:Identifies the  gene.
                SOURCE_EXPRESSION_SYSTEM:Scientific name of the organism in  which the molecule was expressed.
                SOURCE_EXPRESSION_SYSTEM_COMMON:Common name of the organism in  which the moleculewas  expressed.
                SOURCE_EXPRESSION_SYSTEM_TAXID:NCBI Taxonomy ID of the organism  used as the expression  system.
                SOURCE_EXPRESSION_SYSTEM_STRAIN:Strain of the organism in which  the molecule        was  expressed.
                SOURCE_EXPRESSION_SYSTEM_VARIANT:Variant of the organism used as the expression  system.
                SOURCE_EXPRESSION_SYSTEM_CELL_LINE:The specific line of cells used as  the expression  system.
                SOURCE_EXPRESSION_SYSTEM_ATCC_NUMBER:Identifies the ATCC number of the  expression system.
                SOURCE_EXPRESSION_SYSTEM_ORGAN:Specific organ which expressed  the molecule.
                SOURCE_EXPRESSION_SYSTEM_TISSUE:Specific tissue which expressed  the molecule.
                SOURCE_EXPRESSION_SYSTEM_CELL:Specific cell type which  expressed the molecule.
                SOURCE_EXPRESSION_SYSTEM_ORGANELLE:Specific organelle which expressed  the molecule.
                SOURCE_EXPRESSION_SYSTEM_CELLULAR_LOCATION:Identifies the location inside or outside the cell  which expressed the molecule.
                SOURCE_EXPRESSION_SYSTEM_VECTOR_TYPE:Identifies the type of vector used,  i.e.,plasmid,  virus, or cosmid.
                SOURCE_EXPRESSION_SYSTEM_VECTOR:Identifies the vector used.
                SOURCE_EXPRESSION_SYSTEM_PLASMID:Plasmid used in the recombinant experiment.
                SOURCE_EXPRESSION_SYSTEM_GENE:Name of the gene used in  recombinant experiment.
                SOURCE_OTHER_DETAILS:Used to present  information on the source which is not  given elsewhere.
            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                SOURCE:One dict which recode the SOURCE information.
            Examples:

            {'SOURCE_Specification_1': {'SOURCE_MOL_ID': '1', 'SOURCE_ORGANISM_SCIENTIFIC': 'ENTEROBACTERIA PHAGE T4;',
             'SOURCE_ORGANISM_TAXID': '10665;', 'SOURCE_CELL': 'LAR_LOCATION: CYTOPLASM;', 'SOURCE_GENE': 'GENE E;',
              'SOURCE_EXPRESSION_SYSTEM_TAXID': '562;', 'SOURCE_EXPRESSION_SYSTEM_STRAIN': 'RR1;',
              'SOURCE_EXPRESSION_SYSTEM_PLASMID': 'PHS1403;', 'SOURCE_EXPRESSION_SYSTEM_GENE': 'T4 LYSOZYME'},
              'SOURCE_Specification_2': {'SOURCE_MOL_ID': '2', 'SOURCE_ORGANISM_SCIENTIFIC': 'ENTEROBACTERIA PHAGE T4;',
               'SOURCE_ORGANISM_TAXID': '10665;', 'SOURCE_CELL': 'LAR_LOCATION: CYTOPLASM;',
               'SOURCE_GENE': 'test_ GENE E;'},
               'SOURCE_Specification_3': {'SOURCE_MOL_ID': '3', 'SOURCE_GENE': 'test_GENE E;',
               'SOURCE_EXPRESSION_SYSTEM_VECTOR_TYPE': 'test_562test_;', 'SOURCE_EXPRESSION_SYSTEM_VECTOR': 'test_RR1test_;',
               'SOURCE_EXPRESSION_SYSTEM_PLASMID': 'test_ PHS1403;', 'SOURCE_EXPRESSION_SYSTEM_GENE': 'test_ T4 LYSOZYME'},
               'pdb_id': '220L', 'SOURCE_Continuation': '22', 'SOURCE_Specification_num': '3'}
            Raises:
                Many items are not apperence in the test file.So the dic on this words is not all.
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        SOURCE = {}
        SOURCE_Continuation = ''
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'SOURCE':
                SOURCE_Continuation = line[8:10].strip()  # get how many lines does it have.
                #  #get the line number
                # if MOL_ID appeared ,COMPND _Specification id+1

                if 'MOL_ID' == line[10:16].strip() or 'MOL_ID' == line[10:17].strip():
                    # tips: because of strip will let the whiteblank away so it is ok to put[10:17]
                    # if it is first,it is[10:16];other case is[10:17]
                    # it is where to put codes in order to divide items in one mode
                    if ('MOL_ID' == line[10:16].strip()):  # it is mol_id 1
                        SOURCE_Specification_temp = 'SOURCE_Specification_1'
                        SOURCE[SOURCE_Specification_temp] = {}
                        SOURCE[SOURCE_Specification_temp]['SOURCE_MOL_ID'] = line[17:19].strip()
                        SOURCE_Specification_num = line[17:19].strip()
                        pass
                    elif ('MOL_ID' == line[10:17].strip()):  # it is mol_id next
                        SOURCE_Specification_temp = 'SOURCE_Specification_' + str(
                            line[18:20].strip())  # put the id_number next the variable
                        SOURCE[SOURCE_Specification_temp] = {}
                        SOURCE[SOURCE_Specification_temp]['SOURCE_MOL_ID'] = line[18:20].strip()
                        SOURCE_Specification_num = line[18:20].strip()
                    pass
                if ('SYNTHETIC' == line[11:20].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_SYNTHETIC'] = line[21:80].strip()
                # 3
                elif ('FRAGMENT' == line[11:19].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_FRAGMENT'] = line[20:80].strip()
                # 4
                elif ('ORGANISM_SCIENTIFIC' == line[11:30].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ORGANISM_SCIENTIFIC'] = line[31:80].strip()
                # 5
                elif ('ORGANISM_COMMON' == line[11:26].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ORGANISM_COMMON'] = line[27:80].strip()
                # 6
                elif ('ORGANISM_TAXID' == line[11:25].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ORGANISM_TAXID'] = line[26:80].strip()
                # 7
                elif ('STRAIN' == line[11:17].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_STRAIN'] = line[18:80].strip()
                # 8
                elif ('VARIANT' == line[11:18].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_VARIANT'] = line[19:80].strip()
                # 9
                elif ('CELL_LINE' == line[11:20].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_CELL_LINE'] = line[21:80].strip()
                # 10
                elif ('ATCC' == line[11:15].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ATCC'] = line[16:80].strip()
                # 11
                elif ('ORGAN' == line[11:16].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ORGAN'] = line[17:80].strip()
                # 12
                elif ('TISSUE' == line[11:17].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_TISSUE'] = line[18:80].strip()
                # 13
                elif ('CELL' == line[11:15].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_CELL'] = line[16:80].strip()
                # 14
                elif ('ORGANELLE' == line[11:20].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_ORGANELLE'] = line[21:80].strip()
                # 15
                elif ('SECRETION' == line[11:20].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_SECRETION'] = line[21:80].strip()
                # 16
                elif ('CELLULAR_LOCATION' == line[11:28].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_CELLULAR_LOCATION'] = line[29:80].strip()
                # 17
                elif ('PLASMID' == line[11:18].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_PLASMID'] = line[19:80].strip()
                # 18
                elif ('GENE' == line[11:15].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_GENE'] = line[16:80].strip()
                # 19
                elif ('EXPRESSION_SYSTEM' == line[11:28].strip() and ":" == line[28].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM'] = line[29:80].strip()
                # 20
                elif ('EXPRESSION_SYSTEM_COMMON' == line[11:35].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_COMMON'] = line[36:80].strip()
                # 21
                elif ('EXPRESSION_SYSTEM_TAXID' == line[11:34].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_TAXID'] = line[35:80].strip()
                # 22
                elif ('EXPRESSION_SYSTEM_STRAIN' == line[11:35].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_STRAIN'] = line[36:80].strip()
                # 23
                elif ('EXPRESSION_SYSTEM_VARIANT' == line[11:35].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_VARIANT'] = line[36:80].strip()
                # 24
                elif ('EXPRESSION_SYSTEM_CELL_LINE' == line[11:38].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_CELL_LINE'] = line[39:80].strip()
                # 25
                elif ('EXPRESSION_SYSTEM_ATCC_NUMBER' == line[11:40].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_ATCC_NUMBER'] = line[41:80].strip()
                # 26
                elif ('EXPRESSION_SYSTEM_ORGAN' == line[11:34].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_ORGAN'] = line[35:80].strip()
                # 27
                elif ('EXPRESSION_SYSTEM_TISSUE' == line[11:35].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_TISSUE'] = line[36:80].strip()
                # 28
                elif ('EXPRESSION_SYSTEM_CELL' == line[11:33].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_CELL'] = line[34:80].strip()
                # 29
                elif ('EXPRESSION_SYSTEM_ORGANELLE' == line[11:38].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_ORGANELLE'] = line[39:80].strip()
                # 30
                elif ('EXPRESSION_SYSTEM_CELLULAR_LOCATION' == line[11:46].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_CELLULAR_LOCATION'] = line[
                                                                                                      47:80].strip()
                # 31
                elif ('EXPRESSION_SYSTEM_VECTOR_TYPE' == line[11:40].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_VECTOR_TYPE'] = line[41:80].strip()
                # 32
                # test it specially
                elif ('EXPRESSION_SYSTEM_VECTOR' == line[11:35].strip() and '_TYPE' != line[35:40].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_VECTOR'] = line[36:80].strip()
                # 33
                elif ('EXPRESSION_SYSTEM_PLASMID' == line[11:36].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_PLASMID'] = line[37:80].strip()
                # 34
                elif ('EXPRESSION_SYSTEM_GENE' == line[11:33].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_EXPRESSION_SYSTEM_GENE'] = line[34:80].strip()

                elif ('OTHER_DETAILS' == line[11:24].strip()):
                    SOURCE[SOURCE_Specification_temp]['SOURCE_OTHER_DETAILS'] = line[25:80].strip()

            # #print(COMPND)
        SOURCE['pdb_id'] = pdb_id
        SOURCE['SOURCE_Continuation'] = SOURCE_Continuation
        SOURCE['SOURCE_Specification_num'] = SOURCE_Specification_num
        #print(SOURCE)

        return SOURCE

    # 5
    def get_keywords_info(self, PDBfile):
        """Get keywords information
        The KEYWDS record contains a set of terms relevant to the entry. Terms in the KEYWDS record provide a simple
        means of categorizing entries and may be used to generate index files. This record addresses some of the
        limitations found in the classification field of the HEADER record. It provides the opportunity to add further
        annotation to the entry in a concise and computer-searchable fashion.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            KEYWDS_Continuation:Allows concatenation of multiple records.
            KEYWDS_Specification:Title of the  experiment.
            KEYWDS_Specification_temp：
        Used functions:
            __load_file:A internal function which can read PDB file .
        Returns:
            KEYWDS:One dict which recode the KEYWDS information.
        Examples:
            {'pdb_id': '220L', 'KEYWDS_Continuation': '22', 'KEYWDS_Specification': 'HYDROLASE, O-GLYCOSYL, T4 LYSOZYME,
            CAVITY MUTANTS, LIGAND BINDING, PROTEIN ENGINEERING, PROTEIN DESIGN bBINDING, pPROTEIN eENGINEERING,
             pPROTEIN dDESIGN bBINDING, pPROTEIN eENGINEERING, pPROTEIN dDESIGN'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        KEYWDS = {}

        KEYWDS_Specification = ''
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'KEYWDS':
                KEYWDS_Continuation = line[8:10].strip()  # get how many lines does it have.The number 9 maybe false
                KEYWDS_Specification_temp = line[10:81].strip()
                if (KEYWDS_Continuation):  # if continuation has number put a blak to connect strings
                    KEYWDS_Specification = KEYWDS_Specification + " " + KEYWDS_Specification_temp
                else:  # for the first time do not need a blank
                    KEYWDS_Specification = KEYWDS_Specification + KEYWDS_Specification_temp
        KEYWDS['pdb_id'] = pdb_id
        KEYWDS['KEYWDS_Continuation'] = KEYWDS_Continuation
        KEYWDS['KEYWDS_Specification'] = KEYWDS_Specification
        #print(KEYWDS)
        return KEYWDS

    # 6
    def get_expdta_info(self, PDBfile):
        """Get EXPDTA  information
            The EXPDTA record presents information about the experiment.
            The EXPDTA record identifies the experimental technique used. This may refer to the type of radiation and
            sample, or include the spectroscopic or modeling technique. Permitted values include:
                X-RAY  DIFFRACTION
                FIBER  DIFFRACTION
                NEUTRON  DIFFRACTION
                ELECTRON  CRYSTALLOGRAPHY
                ELECTRON  MICROSCOPY
                SOLID-STATE  NMR
                SOLUTION  NMR
                SOLUTION  SCATTERING
            *Note:Since October 15, 2006, theoretical models are no longer accepted for deposition.
            Any theoretical models deposited prior to this date are archived at
            ftp://ftp.wwpdb.org/pub/pdb/data/structures/models.
            Please see the documentation from previous versions for the related file format description.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                EXPDTA_Continuation:Allows concatenation of multiple records.count how many lines.
                EXPDTA_technique:
                EXPDTA_technique_temp:do not need
                EXPDTA_X-RAY  :DIFFRACTION
                EXPDTA_FIBER :DIFFRACTION
                EXPDTA_NEUTRON :DIFFRACTION
                EXPDTA_ELECTRON:MICROSCOPY、CRYSTALLOGRAPHY
                EXPDTA_SOLID-STATE:NMR
                EXPDTA_SOLUTION:SCATTERING、NMR
            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                EXPDTA:One dict which recode the EXPDTA information.
            Examples:
                {'EXPDTA_technique': {'EXPDTA_X_RAY': 'DIFFRACTION'}, 'pdb_id': '220L', 'EXPDTA_Continuation': '0',
                'EXPDTA_technique_num': '1'}
            Raises:
                'EXPDTA_ELECTRON 'and 'EXPDTA_SOLUTION'have many other techniques
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        EXPDTA = {}
        EXPDTA_Continuation = '0'
        EXPDTA_technique_num = '1'
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'EXPDTA':
                # EXPDTA_Continuation = line[8:10].strip()  # get how many lines does it have.#get the line number
                # if MOL_ID appeared ,COMPND _Specification id+1
                # if 'MOL_ID' == line[10:16].strip() or 'MOL_ID' == line[10:17].strip():
                # tips: because of strip will let the whiteblank away so it is ok to put[10:17]
                # if it is first,it is[10:16];other case is[10:17]
                # it is where to put codes in order to divide items in one mode
                # if ('MOL_ID' == line[10:16].strip()):  # it is mol_id 1
                # 	EXPDTA_technique_temp = 'EXPDTA_Specification_1'
                # 	EXPDTA[EXPDTA_technique_temp] = {}
                # 	EXPDTA[EXPDTA_technique_temp]['EXPDTA_MOL_ID'] = line[17:19].strip()
                # 	EXPDTA_technique_num = line[17:19].strip()  # if there is only 1 mol.
                # 	pass
                # elif ('MOL_ID' == line[10:17].strip()):  # it is mol_id next
                # 	EXPDTA_technique_temp = 'EXPDTA_Specification_' + str(
                # 		line[18:20].strip())  # put the id_number next the variable
                # 	EXPDTA[EXPDTA_technique_temp] = {}
                # 	EXPDTA[EXPDTA_technique_temp]['EXPDTA_MOL_ID'] = line[18:20].strip()
                # 	EXPDTA_technique_num = line[18:20].strip()
                # pass
                # I do not delate these codes because of afraid of different sutiations
                EXPDTA_technique_temp = 'EXPDTA_technique'
                EXPDTA[EXPDTA_technique_temp] = {}
                if ('X-RAY' == line[10:15].strip()):
                    # #print(line[17:80].strip())
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_X_RAY'] = line[16:80].strip()
                elif ('FIBER ' == line[10:15].strip()):
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_FIBER'] = line[16:80].strip()
                elif ('NEUTRON' == line[10:17].strip()):
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_NEUTRON '] = line[18:80].strip()
                elif ('ELECTRON ' == line[10:18].strip()):
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_ELECTRON'] = line[19:80].strip()
                elif ('SOLID-STATE' == line[10:21].strip()):
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_SOLID-STATE'] = line[22:80].strip()
                elif ('SOLUTION' == line[10:18].strip()):
                    EXPDTA[EXPDTA_technique_temp]['EXPDTA_SOLUTION'] = line[19:80].strip()

            # #print(COMPND)
        EXPDTA['pdb_id'] = pdb_id
        EXPDTA['EXPDTA_Continuation'] = EXPDTA_Continuation
        EXPDTA['EXPDTA_technique_num'] = EXPDTA_technique_num
        #print(EXPDTA)
        return EXPDTA

    # 7
    def get_author_info(self, PDBfile):
        """Get  author information
        TThe AUTHOR record contains the names of the people responsible for the contents of the entry.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            AUTHOR_Continuation:Allows concatenation of multiple records.
            AUTHOR_Specification:Title of the  experiment.
            AUTHOR_Specification_temp：
        Used functions:
            __load_file:A internal function which can read PDB file .
        Returns:
            AUTHOR:One dict which recode the AUTHORinformation.
        Examples:
            {'pdb_id': '220L', 'AUTHOR_Continuation': 1, 'AUTHOR_authorlist': 'E.P.BALDWIN,W.A.BAASE,
            X.-J.ZHANG,V.FEHER,B.W.MATTHEWS'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        AUTHOR = {}
        AUTHOR_authorlist = ''
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'AUTHOR':
                AUTHOR_Continuation = line[8:10].strip()  # get how many lines does it have.The number 9 maybe false
                AUTHOR_authorlist_temp = line[10:81].strip()
                if (AUTHOR_Continuation):  # if continuation has number put a blak to connect strings
                    AUTHOR_authorlist = AUTHOR_authorlist + " " + AUTHOR_authorlist_temp
                else:  # for the first time do not need a blank
                    AUTHOR_authorlist = AUTHOR_authorlist + AUTHOR_authorlist_temp
                    AUTHOR_Continuation = 1
        AUTHOR['pdb_id'] = pdb_id
        AUTHOR['AUTHOR_Continuation'] = AUTHOR_Continuation
        AUTHOR['AUTHOR_authorlist'] = AUTHOR_authorlist
        #print(AUTHOR)
        return AUTHOR

    # 8
    def get_revdat_info(self, PDBfile):
        """Get REVDAT information
            REVDAT records contain a history of the modifications made to an entry since its release.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                REVDAT_modNum:Modification number.
                REVDAT_Continuation:Allows concatenation of multiple records.
                REVDAT_modDat:Date of modification (or release  for  new entries)  in DD-MMM-YY format. This is not repeated on continued lines.
                REVDAT_modId:ID code of this entry. This is not repeated onContinuation lines.
                REVDAT_modType:An integer identifying the type of   modification. For all  revisions, the modification type is listed as 1
                REVDAT_record:Modification detail.
            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                REVDAT:One dict which recode the REVDAT information.
            Examples:
                {'REVDAT_3': {'REVDAT_modDat': '24-FEB-09', 'REVDAT_modId': '220L', 'REVDAT_modType': '1',
                'REVDAT_record': 'VERSN'}, 'REVDAT_2': {'REVDAT_modDat': '01-APR-03', 'REVDAT_modId': '220L',
                 'REVDAT_modType': '1', 'REVDAT_record': 'JRNL'}, 'REVDAT_1': {'REVDAT_modDat': '18-MAR-98',
                 'REVDAT_modId': '220L', 'REVDAT_modType': '0', 'REVDAT_record': ''}, 'pdb_id': '220L'}
            Raises:
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        REVDAT = {}
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            REVDAT_temp = 'REVDAT_'  # initialization the variable again
            if header == 'REVDAT':
                REVDAT_temp = REVDAT_temp + str(line[8:10].strip())
                REVDAT[REVDAT_temp] = {}
                REVDAT[REVDAT_temp]['REVDAT_modDat'] = line[13:22].strip()
                REVDAT[REVDAT_temp]['REVDAT_modId'] = line[23:27].strip()
                REVDAT[REVDAT_temp]['REVDAT_modType'] = line[30:32].strip()
                REVDAT[REVDAT_temp]['REVDAT_record'] = line[33:80].strip()
        REVDAT['pdb_id'] = pdb_id
        #print(REVDAT)
        return REVDAT

    # 9
    def get_remark2_info(self, PDBfile):
        """Get REMARK2 information
            REMARK 2 states the highest resolution, in Angstroms, that was used in building the model.
            As with all the remarks, the first REMARK 2 record is empty and is used as a spacer.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                REMARK_logo:Modification number.
                REMARK2_RESOLUTION:
            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                REMARK2:One dict which recode the REMARK2 information.
            Examples:
                {{'RESOLUTION': '12345.678ANGSTROMS.', 'REMARK_logo': '2', 'pdb_id': '220L'}
                {'RESOLUTION': 'NOT APPLICABLE','REMARK_logo': '2', 'pdb_id': '220L'}
            Raises:
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        REMARK2 = {}
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'REMARK':
                if str(2) == line[8:10].strip():
                    if 'RESOLUTION' == line[11:21].strip():
                        #print()
                        REMARK2['RESOLUTION'] = line[22:80].strip()
        REMARK2['REMARK_logo'] = '2'
        REMARK2['pdb_id'] = pdb_id
        #print(REMARK2)
        return REMARK2

    # 10
    def get_remark3_info(self, PDBfile):
        """Get REMARK2 information
            REMARK 3 presents information on refinement program(s) used and related statistics.
            For non-diffraction studies, REMARK 3 is used to describe any refinement done,
            but its format is mostly free text.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                REMARK_logo:Modification number.
                REMARK3_PROGRAM:
                REMARK3_AUTHORS:
                REMARK3_FREETEXT:
                REMARK3_TITLEtemp:
                REMARK3_FREETEXTtemp:

            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                REMARK3:One dict which recode the REMARK3 information.
            Examples:
                {'PROGRAM': 'TNT 2TNT4 3TNT5 ',
                'AUTHORS': 'TRONRUD,TEN EYCK,MATTHEWS RUD,TEN EYCK,MATTHEWS2 3TRONRUD',
                'DATA USED IN REFINEMENT.': 'RESOLUTION RANGE HIGH (ANGSTROMS) : 1.85    RESOLUTION RANGE LOW
                (ANGSTROMS) : 10.00    DATA CUTOFF            (SIGMA(F)) : 0.000    COMPLETENESS FOR RANGE        (%) : 72.0
                NUMBER OF REFLECTIONS             : 13746        ',
                'NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.': 'PROTEIN ATOMS: 1289NUCLEIC ACID ATOMS   : 0HETEROGEN ATOMS
                      : 2SOLVENT ATOMS: 126',
                 'WILSON B VALUE (FROM FCALC,A**2) : 34.100': '',
                 'RMS DEVIATIONS FROM IDEAL VALUES.RMSWEIGHT  COUNT': 'BOND LENGTHS (A) : 0.015 ; NULL  ;
                  NULLBOND ANGLES(DEGREES) : 2.100 ; NULL  ; NULLTORSION ANGLES (DEGREES) : NULL  ; NULL  ;
                   NULLPSEUDOROTATION ANGLES  (DEGREES) : NULL  ; NULL  ; NULLTRIGONAL CARBON PLANES   (A) : 0.008 ;
                   NULL  ; NULLGENERAL PLANES   (A) : 0.014 ; NULL  ; NULLISOTROPIC THERMAL FACTORS (A**2) : 5.990 ; NULL  ;
                   NULLNON-BONDED CONTACTS  (A) : 0.049 ; NULL  ; NULL',
                 'INCORRECT CHIRAL-CENTERS (COUNT) : NULL': '',
                 'BULK SOLVENT MODELING.': 'METHOD USED : BABINET SCALINGKSOL: 0.55BSOL: 96.00',
                 'RESTRAINT LIBRARIES.': 'STEREOCHEMISTRY : TNT PROTGEOISOTROPIC THERMAL FACTOR RESTRAINTS : NULL',
                 'OTHER REFINEMENT REMARKS: NULL': '',
                 'REMARK_logo': '3',
                 'pdb_id': '220L'}

            Raises:
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        REMARK3 = {}
        REMARK3_FREETEXT = ''
        REMARK3_FREETEXTtemp = ''
        REMARK3_TITLEtemp = ''
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'REMARK':
                if str(3) == line[8:10].strip():
                    # under this line we step into REMARK3
                    ##print(line[13:20])
                    if 'PROGRAM' == line[13:20].strip():  # when it comes program ,change the title
                        REMARK3_TITLEtemp = 'PROGRAM'
                        REMARK3_FREETEXTtemp = ''
                        REMARK3_FREETEXT = line[27:80].strip() + " "
                    elif 'AUTHORS' == line[13:20].strip():  # when it comes authors ,change the title
                        REMARK3[REMARK3_TITLEtemp] = REMARK3_FREETEXT         #if program have only one line,it need to put the value into the dic.
                        REMARK3_TITLEtemp = 'AUTHORS'
                        REMARK3_FREETEXTtemp = ''
                        REMARK3_FREETEXT = line[27:80].strip() + " "
                    elif '' == line[13:20].strip() and line[27:80].strip():
                        # when it comes blank and have words after do not change the title
                        REMARK3_FREETEXTtemp = line[27:80].strip() + " "
                        REMARK3_FREETEXT = REMARK3_FREETEXT + REMARK3_FREETEXTtemp
                        REMARK3[REMARK3_TITLEtemp] = REMARK3_FREETEXT
                    elif " " != line[12] and 'REFINEMENT' != line[11:21].strip():
                        # when comes here there is no more author and program and we get the outstanding line as the key
                        REMARK3_TITLEtemp1 = line[12:80].strip()
                        #change REMARK3_TITLEtemp's '.' into '_' because key must not contain it .
                        REMARK3_FREETEXTtemp = ''
                        REMARK3_FREETEXT = ''
                        REMARK3_TITLEtemp=REMARK3_TITLEtemp1.replace('.','_')
                        if REMARK3_TITLEtemp != '':
                            REMARK3[REMARK3_TITLEtemp] = REMARK3_FREETEXT
                    # it is for the last line espically it have only one line
                    elif ' ' == line[12] and ' ' != line[13].strip():
                        # we connected the lines and add them to dic here
                        REMARK3_FREETEXTtemp = line[13:80].strip() + "    "  # each line add four space
                        REMARK3_FREETEXT = REMARK3_FREETEXT + REMARK3_FREETEXTtemp
                        if REMARK3_TITLEtemp != '':
                            REMARK3[REMARK3_TITLEtemp] = REMARK3_FREETEXT
        REMARK3['REMARK_logo'] = '3'
        REMARK3['pdb_id'] = pdb_id
        #print(REMARK3)
        return REMARK3

    # 11
    def get_seqres_info(self, PDBfile):
        """Get SEQRES information
            SEQRES records contain a listing of the consecutive chemical components covalently linked in a linear
            fashion to form a polymer. The chemical components included in this listing may be standard or modified
            amino acid and nucleic acid residues. It may also include other residues that are linked to the standard
            backbone in the polymer. Chemical components or groups covalently linked to side-chains (in peptides) or
            sugars and/or bases (in nucleic acid polymers) will not be listed here.
            Args:
                PDBfile:the full path of PDB to be parsed
            Variables(attributes):
                pdb_id: the ID of the PDBfile which is to be parsed
                line: The line to be parsed
                SEQRES_serNum:Serial number of the SEQRES record for  the current  chain. Starts at 1 and increments
                                by one  each line. Reset to 1 for each chain.
                SEQRES_chainID:Chain identifier. This may be any single legal  character, including a blank
                                which is is  used if there is only one chain.
                SEQRES_numRes:Number of residues in the chain.This  value is repeated on every record.
                SEQRES_resName:it is a long string
                SEQRES_seqres:
            Used functions:
                __load_PDB_file:A internal function which can read PDB file .
            Returns:
                SEQRES:One dict which recode the SEQRES information.
            Examples:
                {'pdb_id': '220L', 'SEQRES_serNum': '17',
                'A': {'SEQRES_seqres': 'MET ASN ILE ...LYS ASN LEU ', 'SEQRES_numRes': '164', 'SEQRES_chainID': 'A'},
                'B': {'SEQRES_seqres': 'ABC DEF ILE ...ILE XXX XXX ', 'SEQRES_numRes': '64', 'SEQRES_chainID': 'B'}}
            Raises:
                'FRAGMENT'is not apperence in the test file.So there is no items named"FTAGMENT in
                IOError:None
            """
        lines = self.__load_PDB_file(PDBfile)
        # define variables
        SEQRES = {}
        SEQRES_serNum = ''
        SEQRES_seqres = ''
        SEQRES_chainID = ''
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'SEQRES':
                SEQRES_serNum = line[8:10].strip()  # get how many lines does it have.The number 9 maybe false
                # get the line number
                if SEQRES_chainID != line[11:12].strip():
                    SEQRES_seqres = ''
                SEQRES_chainID = line[11:12].strip()
                SEQRES_numRes = line[14:17].strip()
                SEQRES_resName = line[19:70].strip() + ' '
                SEQRES_seqres = SEQRES_seqres + SEQRES_resName
                SEQRES['pdb_id'] = pdb_id
                SEQRES['SEQRES_serNum'] = SEQRES_serNum
                SEQRES[SEQRES_chainID] = {}
                SEQRES[SEQRES_chainID]['SEQRES_seqres'] = SEQRES_seqres
                SEQRES[SEQRES_chainID]['SEQRES_numRes'] = SEQRES_numRes
                SEQRES[SEQRES_chainID]['SEQRES_chainID'] = SEQRES_chainID
        #print(SEQRES)
        return SEQRES

    # 12
    def get_cryst1_info(self, PDBfile):
        """Get cryst1 information.
        The CRYST1 record presents the unit cell parameters, space group, and Z value. If the structure was not
        determined by crystallographic means, CRYST1 simply provides the unitary values, with an appropriate REMARK.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            CRYST1_a:a (Angstroms).
            CRYST1_b:b (Angstroms).
            CRYST1_c:c (Angstroms).
            CRYST1_alpha:alpha (degrees).
            CRYST1_beta:beta (degrees).
            CRYST1_gamma:gamma (degrees).
            CRYST1_sGroup:Space  group.
            CRYST1_z:Z value.
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
        Returns:
            CRYST1:One dict which recode the CRYST1 information.
        Examples:
            {'CRYST1_a': '60.900', 'CRYST1_b': '60.900', 'CRYST1_c': '97.200',
            'CRYST1_alpha': '90.00', 'CRYST1_beta': '90.00', 'CRYST1_gamma': '120.00',
            'CRYST1_sGroup': '32 2 1', 'CRYST1_z': '6', 'pdb_id': '220L｝
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        CRYST1 = {}
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'CRYST1':
                CRYST1_a = line[7:15].strip()
                CRYST1_b = line[16:24].strip()
                CRYST1_c = line[25:33].strip()
                CRYST1_alpha = line[34:40].strip()
                CRYST1_beta = line[41:47].strip()
                CRYST1_gamma = line[48:54].strip()
                CRYST1_sGroup = line[55:66].strip()
                CRYST1_z = line[67:70].strip()
        # put key_values for dic
        CRYST1['CRYST1_a'] = CRYST1_a
        CRYST1['CRYST1_b'] = CRYST1_b
        CRYST1['CRYST1_c'] = CRYST1_c
        CRYST1['CRYST1_alpha'] = CRYST1_alpha
        CRYST1['CRYST1_beta'] = CRYST1_beta
        CRYST1['CRYST1_gamma'] = CRYST1_gamma
        CRYST1['CRYST1_sGroup'] = CRYST1_sGroup
        CRYST1['CRYST1_z'] = CRYST1_z
        CRYST1['pdb_id'] = pdb_id

        #print(CRYST1)
        return CRYST1

    # 13
    def get_origxn_info(self, PDBfile):
        """Get ORIGXn information.
        The ORIGXn (n = 1, 2, or 3) records present the transformation from the orthogonal coordinates contained
        in the entry to the submitted coordinates.
        Args:
           PDBfile:the full path of PDB to be parsed
        Variables(attributes):
           pdb_id: the ID of the PDBfile which is to be parsed
           line: The line to be parsed
           ORIGXn_o1:o[n][1]
           ORIGXn_o2:o[n][2]
           ORIGXn_o3:o[n][3]
           ORIGXn_tn:t[n]
        Used functions:
           __load_PDB_file:A internal function which can read PDB file .
        Returns:
           ORIGXN:One dict which recode the ORIGXN information.
        Examples:
        {'ORIGX1': {'ORIGXN_o1': '1.000000', 'ORIGXN_o2': '0.000000', 'ORIGXN_o3': '0.000000', 'ORIGXN_tn': '0.00000'},
         'ORIGX2': {'ORIGXN_o1': '0.000000', 'ORIGXN_o2': '1.000000', 'ORIGXN_o3': '0.000000', 'ORIGXN_tn': '0.00000'},
         'ORIGX3': {'ORIGXN_o1': '0.000000', 'ORIGXN_o2': '0.000000', 'ORIGXN_o3': '1.000000', 'ORIGXN_tn': '0.00000'},
         'pdb_id': '220L'}
        Raises:
           IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        ORIGXN = {}
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line[0:5]
            if header == 'HEADE':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'ORIGX':
                ORIGXN[str(line[0:6])] = {}
                ORIGXN_o1 = line[11:20].strip()
                ORIGXN_o2 = line[21:30].strip()
                ORIGXN_o3 = line[31:40].strip()
                ORIGXN_tn = line[46:55].strip()
                ORIGXN[line[0:6]]['ORIGXN_o1'] = ORIGXN_o1
                ORIGXN[line[0:6]]['ORIGXN_o2'] = ORIGXN_o2
                ORIGXN[line[0:6]]['ORIGXN_o3'] = ORIGXN_o3
                ORIGXN[line[0:6]]['ORIGXN_tn'] = ORIGXN_tn
        ORIGXN['pdb_id'] = pdb_id
        #print(ORIGXN)
        return ORIGXN

    # 14
    def get_scalen_info(self, PDBfile):
        """Get SCALEn information.
        The SCALEn (n = 1, 2, or 3) records present the transformation from the orthogonal coordinates as contained in
        the entry to fractional crystallographic coordinates. Non-standard coordinate systems should be explained in
         the remarks.
        Args:
           PDBfile:the full path of PDB to be parsed
        Variables(attributes):
           pdb_id: the ID of the PDBfile which is to be parsed
           line: The line to be parsed
           SCALEn_s1:s[n][1]
           SCALEn_s2:s[n][2]
           SCALEn_s3:s[n][3]
           SCALEn_un:u[n]
        Used functions:
           __load_PDB_file:A internal function which can read PDB file .
        Returns:
           SCALEN:One dict which recode the SCALEN information.
        Examples:
        {'SCALE1': {'SCALEN_s1': '0.016420', 'SCALEN_s2': '0.009480', 'SCALEN_s3': '0.000000', 'SCALEN_un': '0.00000'},
         'SCALE2': {'SCALEN_s1': '0.000000', 'SCALEN_s2': '0.018961', 'SCALEN_s3': '0.000000', 'SCALEN_un': '0.00000'},
          'SCALE3': {'SCALEN_s1': '0.000000', 'SCALEN_s2': '0.000000', 'SCALEN_s3': '0.010288', 'SCALEN_un': '0.00000'},
           'pdb_id': '220L'}
        Raises:
           IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        SCALEN = {}
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line[0:5]
            if header == 'HEADE':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'SCALE':
                SCALEN[str(line[0:6])] = {}
                SCALEN_s1 = line[11:20].strip()
                SCALEN_s2 = line[21:30].strip()
                SCALEN_s3 = line[31:40].strip()
                SCALEN_un = line[46:55].strip()
                SCALEN[line[0:6]]['SCALEN_s1'] = SCALEN_s1
                SCALEN[line[0:6]]['SCALEN_s2'] = SCALEN_s2
                SCALEN[line[0:6]]['SCALEN_s3'] = SCALEN_s3
                SCALEN[line[0:6]]['SCALEN_un'] = SCALEN_un
        SCALEN['pdb_id'] = pdb_id
        #print(SCALEN)
        return SCALEN

    # 15
    def get_master_info(self, PDBfile):
        """Get cryst1 information.
            The MASTER record is a control record for bookkeeping. It lists the number of lines in the coordinate entry
            or file for selected record types. MASTER records only the first model when there are multiple models in the
            coordinates.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            MASTER_numRemark:Number of REMARK records
            MASTER_numHet:Number of HET records
            MASTER_numHelix:Number of HELIX records
            MASTER_numSheet:Number of SHEET records
            MASTER_numTurn:deprecated
            MASTER_numSite:Number of SITE records
            MASTER_numXform:Number of coordinate transformation records  (ORIGX+SCALE+MTRIX)
            MASTER_numCoord:Number of atomic coordinate records records (ATOM+HETATM)
            MASTER_numTer:Number of TER records
            MASTER_numConect:Number of CONECT records
            MASTER_numSeq:Number of SEQRES records
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
        Returns:
            CRYST1:One dict which recode the CRYST1 information.
        Examples:
            {'MASTER_numRemark': '317', 'MASTER_0': '0', 'MASTER_numHet': '5', 'MASTER_numHelix': '10',
            'MASTER_numSheet': '2', 'MASTER_numTurn': '0', 'MASTER_numSite': '6', 'MASTER_numXform': '6',
             'MASTER_numCoord': '1417', 'MASTER_numTer': '1', 'MASTER_numConect': '14', 'MASTER_numSeq': '13',
             'pdb_id': '220L'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        # define variables
        MASTER = {}
        for g in range(0, len(lines)):
            line = lines[g]
            # #print(line)
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            if header == 'MASTER':
                MASTER_numRemark = line[10:15].strip()
                MASTER_0 = line[15:20].strip()
                MASTER_numHet = line[20:25].strip()
                MASTER_numHelix = line[25:30].strip()
                MASTER_numSheet = line[30:35].strip()
                MASTER_numTurn = line[35:40].strip()
                MASTER_numSite = line[40:45].strip()
                MASTER_numXform = line[45:50].strip()
                MASTER_numCoord = line[50:55].strip()
                MASTER_numTer = line[55:60].strip()
                MASTER_numConect = line[60:65].strip()
                MASTER_numSeq = line[65:70].strip()
        # put key_values for dic
        MASTER['MASTER_numRemark'] = MASTER_numRemark
        MASTER['MASTER_0'] = MASTER_0
        MASTER['MASTER_numHet'] = MASTER_numHet
        MASTER['MASTER_numHelix'] = MASTER_numHelix
        MASTER['MASTER_numSheet'] = MASTER_numSheet
        MASTER['MASTER_numTurn'] = MASTER_numTurn
        MASTER['MASTER_numSite'] = MASTER_numSite
        MASTER['MASTER_numXform'] = MASTER_numXform
        MASTER['MASTER_numCoord'] = MASTER_numCoord
        MASTER['MASTER_numTer'] = MASTER_numTer
        MASTER['MASTER_numConect'] = MASTER_numConect
        MASTER['MASTER_numSeq'] = MASTER_numSeq
        MASTER['pdb_id'] = pdb_id
        #print(MASTER)
        return MASTER
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #from gongjt
    #16
    def get_site_info(self, PDBfile):
        """Get site information.
            This function can find informations about site Further details on important sites of the entry.  REMARK 800 is mandatory if SITE records exist.
            SITE:Identification of groups comprising important entity sites.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBfile which is to be parsed
            line: The line to be parsed
            Sites: The dict that record the sites information, key is residue
                   position and value is residue abbreviation.
            hetbindingsite: The dict that record sites description, which chain to
                   bind and the binding sites names
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
        Returns:
            totalsite: The dict that record the DB references in the PDBFile, which is  from PDBFile's lines remarked 'DBREF'. such as UNP, PDB DB
        Examples:
            {'1KMH_A': [{'position': {'51': 'G', '65': 'L', '131': 'E', '274': 'M', '297': 'R'}, 'site_description': 'RESIDUE TTX B 499'}],
            '1KMH_B': [{'position': {'81': 'A', '82': 'T', '83': 'D'}, 'site_description': 'RESIDUE TTX B 499'}]}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        # #print (lines)
        Sites = {}
        hetbindingsite = {}
        for g in range(0, len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            elif header == "SITE":
                self.__parseSiteline(line, Sites, pdb_id)
		#print(Sites)
            elif line[:27].upper() == 'REMARK 800 SITE_IDENTIFIER:':
                self.__parseSiteDescripLine(lines, g, hetbindingsite)
		#print(hetbindingsite)

		
        totalsite = self.__formatSiteStructure(Sites, hetbindingsite)

        return totalsite
    #17
    def get_sequence_fromATOM(self, PDBfile):
        """Get sequence from 'ATOM' lines in PDBfile which showed primary structure(s).
            This function can show the sequence which not all of the total sequence ,but what it shown have 3d structures.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            PDBfile: the full path of PDB to be parsed
            pdb_id: the ID of the PDBFile which is to be parsed
            line: The line to be parsed
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
            __parse_PDB_ID_Line:get PDB ID
            __parsePriLine:
            __formatPriStructure
        Returns:
            priStructure: The dict that record the primary structure(s) in the PDBFile ,
                    which is from PDBFile's lines remarked 'ATOM'
        Examples:
            {'1KMH_A': 'KVVN...MERF', '1KMH_B': 'NLGR...EATA'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        primaryStr = {}
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            elif header == 'ATOM':
                self.__parsePriLine(line, primaryStr, pdb_id)
        priStructure = self.__formatPriStructure(primaryStr)
        #print(priStructure)
        return priStructure

    def get_atom_info(self, PDBfile):
        """Get 3Dstructures from 'ATOM' lines in PDB file which shows tertiary structure(s).
            This function can get the tertiary structure(s).It return a very complicated structure and have many items,you can see it clearly in Examples below.
            It contain: Protian ID;chain name;chain langth(begin to end);residues;residue name;residue index;atoms name;atom coordinates(x,y,z);atom_tempfactor.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBFile which is to be parsed
            line: The line to be parsed
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
            __parse_PDB_ID_Line:get PDB ID .
            __parseTerLine:
            __formatTerStructure:
        Returns:
            terStructure: The dict that record the tertiary structure(s)
                    which is from PDBFile's lines remarked 'ATOM'
        Examples:
            {'1KMH_A':
                {'chain_name': '1KMH_A',
                 'chain_langth': {'sequence_begin': '25', 'sequence_end': '501'},
                 'residues': [{'residue_name': 'K',
                               'residue_index': '25',
                               'atoms': [{'atom_name': 'N', 'atom_coordinates': ['47.133', '43.402', '85.802'], 'atom_tempfactor': '186.45'},
                                         {'atom_name': 'CA', 'atom_coordinates': ['46.945', '42.063', '85.139'], 'atom_tempfactor': '186.94'},
                                         {'atom_name': 'C', 'atom_coordinates': ['47.904', '41.771', '83.969'], 'atom_tempfactor': '185.59'},
                                         {'atom_name': 'O', 'atom_coordinates': ['47.639', '42.126', '82.819'], 'atom_tempfactor': '185.59'},
                                         {'atom_name': 'CB', 'atom_coordinates': ['45.514', '41.915', '84.637'], 'atom_tempfactor': '187.53'},
                                         {'atom_name': 'CG', 'atom_coordinates': ['44.493', '41.806', '85.742'], 'atom_tempfactor': '189.44'},
                                         {'atom_name': 'CD', 'atom_coordinates': ['43.087', '41.600', '85.188'], 'atom_tempfactor': '189.23'},
                                         {'atom_name': 'CE', 'atom_coordinates': ['42.097', '41.225', '86.306'], 'atom_tempfactor': '187.48'},
                                         {'atom_name': 'NZ', 'atom_coordinates': ['42.464', '39.974', '87.065'], 'atom_tempfactor': '183.58'}]},
                              {'residue_name': 'V',
                               'residue_index': '26',
                               'atoms': [{'atom_name': 'N', 'atom_coordinates': ['49.054', '41.191', '84.275'], 'atom_tempfactor': '183.54'},
                                        {'atom_name': 'CA', 'atom_coordinates': ['49.967', '40.759', '83.251'], 'atom_tempfactor': '181.80'},
                                        {'atom_name': 'C', 'atom_coordinates': ['49.435', '39.550', '82.503'], 'atom_tempfactor': '177.17'},
                                        {'atom_name': 'O', 'atom_coordinates': ['50.224', '38.821', '81.887'], 'atom_tempfactor': '178.17'},
                                        {'atom_name': 'CB', 'atom_coordinates': ['51.312', '40.398', '83.865'], 'atom_tempfactor': '184.04'},
                                        {'atom_name': 'CG1', 'atom_coordinates': ['52.403', '40.385', '82.794'], 'atom_tempfactor': '186.11'},
                                        {'atom_name': 'CG2', 'atom_coordinates': ['51.662', '41.379', '84.993'], 'atom_tempfactor': '186.44'}]}, ...]}}

        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        tertiarystr = []
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            elif header == 'ATOM':
                self.__parseTerLine(line, tertiarystr, pdb_id)
        terStructure = self.__formatTerStructure(tertiarystr)
        #print(terStructure)
        return terStructure

    def get_reference_betweenPDB(self, PDBfile):
        """Get reference between PDB and UNP.
            This function can get the reference from PDB file ,so that you can find the protain in other DB.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            pdb_id: the ID of the PDBFile which is to be parsed
            line: The line to be parsed
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
            __parse_PDB_ID_Line:get PDB ID .
            __parseDBrefline:
            __formatDBref:
        Returns:
            DBREF: The dict that record the DB references in the PDBFile, which
               is  from PDBFile's lines remarked 'DBREF'. such as UNP, PDB DB
        Examples:
            {'1KMH_A': {'IDs': {'PDB_chain_ID': '1KMH_A', 'UNP': ['P06450']}},
             '1KMH_B': {'IDs': {'PDB_chain_ID': '1KMH_B', 'UNP': ['P00825']}}}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        DBref = {}
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            elif header == "DBREF":
                self.__parseDBrefline(line, DBref)
        DBREF = self.__formatDBref(DBref)
        #print(DBREF)
        return DBREF

    def get_sequence_fromSEQ(self, PDBfile):
        """Get sequence from 'SEQ' lines in PDBfile which showed the whole protein chain .
	        This function can show the sequence of the total sequence and it is longer than the sequence get from ATOM .
	        Some of the residue did not get the 3D structure,so in "ATOM",the sequence is half-baked.
	        These two sequence must be the same somewhere.
        Args:
            PDBfile:the full path of PDB to be parsed
        Variables(attributes):
            PDBfile: the full path of PDB to be parsed
            pdb_id: the ID of the PDBFile which is to be parsed
            line: The line to be parsed
        Used functions:
            __load_PDB_file:A internal function which can read PDB file .
            __parse_PDB_ID_Line:get PDB ID
            __parseSeqresLine:
        Returns:
            seqres: The dict that record the primary structure(s) in the PDBFile, which
                is  from PDBFile's lines remarked 'SEQRES'
        Examples:
            {'1KMH_A': 'MAT...EQA',
            '1KMH_B': 'MRI...LKK'}
        Raises:
            IOError:None
        """
        lines = self.__load_PDB_file(PDBfile)
        seqres = {}
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':
                pdb_id = self.__parse_PDB_ID_Line(line)
            elif header == "SEQRES":
                self.__parseSeqresLine(line, seqres, pdb_id)
        #print(seqres)
        return seqres

    """
    get site information in the ‘SITE’ line
    """
    def __parseSiteline(self, line, Sites, pdb_id):
        """ @return: None
	        @param
	        line: The line to be parsed
	        Sites： A dict that record Sites from PDBfile
	        chain: The dict that record one line site information such as
               some residues in Site(s) and their position in PDBfile
        """
        chainid = pdb_id + "_"
        site_name = line[11:14].strip()
        chain = {}
    
        for eachone in [22, 33, 44, 55]:
            chainone = line[eachone]
            pdbid_chainid = chainid + line[eachone]
            site_res = line[eachone - 4:eachone - 1].strip()
            site_pos = line[eachone + 1:eachone + 5].strip()
            # #print(chain)
            if chainone != ' ' or site_res != '':
                if site_res == 'HOH':
                    pass
                else:
                    if pdbid_chainid not in chain:
                        chain.update({pdbid_chainid: {site_pos: self.__transAA(site_res)}})
                    else:
                        chain[pdbid_chainid].update({site_pos: self.__transAA(site_res)})
        # #print(chain)
        for key in chain:
            if key not in Sites:
                Sites.update({key: {site_name: chain[key]}})
            else:
                if site_name in Sites[key]:
                    Sites[key][site_name].update(chain[key])
                else:
                    Sites[key].update({site_name: chain[key]})
        return None

    def __parse_PDB_ID_Line(self, line):
        """Give the protein an ID.
            This function get Protein when somewhere need an ID .
        Args:
            line: The line to be parsed.[62:66]
        Variables(attributes):
            None
        Used functions:
            None
        Returns:
            chainid :the unique id of the protein .
        Examples:
            "1KMH"
        Raises:
            IOError:None
        """
        chainid = line[62:66].strip()  # 返回第62-66位上的内容
        return chainid

    def __parseSiteDescripLine(self, lines, g, hetbindingsite):
        """
	        @return: None
	        @param
	        lines: all lines in PDBfile
	        g: The index of the line to be parsed.
	        hetbindingsite: The dict that record sites description, which chain to
	            bind and the binding sites names
        """
        line = lines[g]
        bindingsite = line.split()[3]
        description = ''
    
        if lines[g + 2][:28].upper() == 'REMARK 800 SITE_DESCRIPTION:':
            if lines[g + 2][28:45].strip().upper() == 'BINDING SITE FOR':
                description = lines[g + 2].rstrip()[46:]
                fulldescription = self.__NextSiteDescripLine(description, lines, g)
            else:
                description = lines[g + 2].rstrip()[29:]
                fulldescription = self.__NextSiteDescripLine(description, lines, g)
        
            hetbindingsite.update({bindingsite: fulldescription})
    
        return None

    def __NextSiteDescripLine(self, description, lines, g):
        """@return: The sites description
	        @param
	        lines: all lines in PDBfile
	        g: The index of the line to be parsed.
        """
        lineindex = g + 3
    
        while lines[lineindex].strip()[:10] == 'REMARK 800' and len(lines[lineindex].strip().split()) > 2:
            if lines[lineindex].strip().split()[2] != "SITE" and lines[lineindex].strip().split()[
                2] != 'SITE_IDENTIFIER:' and lines[lineindex].strip().split()[
                2] != 'EVIDENCE_CODE:' and lines[lineindex].strip().split()[
                2] != 'SITE_DESCRIPTION:' :
                description = description + lines[lineindex].strip()[11:]
            lineindex = lineindex + 1
    
        return description

    def __transAA(self, x):
        """@return: The one-character amino acid name
	        @param  x: The nonstandard amino acid name
        """
        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        if x in d.keys():
            return d[x]
        else:
            return 'X'

    def __formatSiteStructure(self, Sites, hetbindingsite):
        """@return: The dict that record some sites and their binding description
	        @param
	        line: The line to be parsed
	        Sites： A dict that record Sites from PDBfile,one site include numerous
	                residues
	        hetbindingsite: The dict that record Het binding some chain description
        """
        totalsite = {}
        for chain in Sites.keys():
            chainsites = []
            for sitename in Sites[chain].keys():
                if sitename in hetbindingsite.keys():
                    chainsites.append({'position': Sites[chain][sitename], "site_description": hetbindingsite[sitename]})
                else:
                    pass
                    #print(sitename)

                totalsite.update({chain: chainsites})
                pass
        return totalsite
        
    def __parsePriLine(self, line, priStructure, pdb_id):
        """
        @return: None
        @param
        line: The line to be parsed
        priStructure: The dict that record the primary structure(s) in the PDBFile
        """
        chain_name = pdb_id + "_" + line[21]
        atom_name = line[12:16].strip()
        residue_name = self.__transAA(line[17:20].strip())
        temp_factor = line[60:66].strip()
        resseq_position = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()
        
        if chain_name in priStructure.keys():
            if int(resseq_position) - int(priStructure[chain_name]['residue_index']) <= 0:
                pass
            else:
                #print(int(resseq_position))
                #print(int(priStructure[chain_name]['residue_index']))
                for j in range(0, (int(resseq_position) - int(priStructure[chain_name]['residue_index'])) - 1):
                    priStructure[chain_name][chain_name] = priStructure[chain_name][chain_name] + "_"
                    #print(priStructure[chain_name][chain_name])
                    #print(line)
		    
                priStructure[chain_name][chain_name] = priStructure[chain_name][chain_name] + residue_name
                priStructure[chain_name]['residue_index'] = resseq_position
        else:
            priStructure.update({chain_name: {chain_name: residue_name, 'residue_index': resseq_position}})
        
        # #print('@\n')
        return None

    def __formatPriStructure(self, primaryStr):
        """
        @return: the normalize dict that primary structure(s) information
        @param
        primaryStr: The dict that record primary structure(s) from PDBfile
        """
        priStructure = {}
        for pdbchain in primaryStr.keys():
            priStructure.update(primaryStr[pdbchain])
        if "residue_index" in priStructure.keys():
            del priStructure["residue_index"]
        else:
            print(priStructure)

    
        return priStructure

    def __parseTerLine(self, line, terStructure, pdb_id):
        """
        @return: None
        @param
        line: The line to be parsed
        priStructure: The dict that record the tertiary structure(s) in the PDBFile
        """
        # #print(line)
        chain_name = pdb_id + "_" + line[21]
        atom_name = line[12:16].strip()
        residue_name = self.__transAA(line[17:20].strip())
        temp_factor = line[60:66].strip()
        resseq_position = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()
    
        label = True
    
        for j in range(0, len(terStructure)):
            if terStructure[j]['chain_name'] == chain_name:
                elelen = len(terStructure[j]['residues'])
                label = False
                count = 0
                for i in range(0, elelen):
                    if terStructure[j]['residues'][i]['residue_name'] == residue_name and \
                                    terStructure[j]['residues'][i]['residue_index'] == resseq_position:
                        terStructure[j]['residues'][i]['atoms'].append(
                            {'atom_name': atom_name, 'atom_coordinates': [x, y, z], 'atom_tempfactor': temp_factor})
                        count = 1
                        break
            
                if count == 0:
                    terStructure[j]['residues'].append(
                        {'residue_name': residue_name, 'residue_index': resseq_position, 'atoms': [
                            {'atom_name': atom_name, 'atom_coordinates': [x, y, z],
                             'atom_tempfactor': temp_factor}]})
                    sequence_end = terStructure[j]['chain_langth']['sequence_end']
                    if int(sequence_end) < int(resseq_position):
                        terStructure[j]['chain_langth']['sequence_end'] = resseq_position
    
        if label == True:
            terStructure.append({'chain_name': chain_name, 'chain_langth': {'sequence_begin': resseq_position,
                                                                            'sequence_end': resseq_position},
                                 'residues': [{'residue_name': residue_name, 'residue_index': resseq_position,
                                               'atoms': [{'atom_name': atom_name, 'atom_coordinates': [x, y, z],
                                                          'atom_tempfactor': temp_factor}]}]})
    
        # #print(terStructure)
        # #print('!')
        return None

    def __formatTerStructure(self, tertiarystr):
        terStructure = {}
        if tertiarystr != []:
            for i in range(0, len(tertiarystr)):
                terStructure[tertiarystr[i]["chain_name"]] = tertiarystr[i]
        else:
            pass
        return terStructure

    def __parseDBrefline(self, line, DBref):
        """
        @return: None
        @param
        line: The line to be parsed
        DBref： A dict that record DB reference information form PDBfile's
              'DBREF' lines
        """
        chainid = line[7:11] + "_" + line[12]
        dbname = line[26:32].strip().upper()
        dbentryid = line[33:41].strip()
        dbentryname = line[42:54].strip()
        pdbchainbeg = line[14:18].strip()
        unichainbeg = line[55:60].strip()
    
        if chainid not in DBref:
            DBref.update({chainid: {"IDs": {'PDB_chain_ID': chainid, dbname: [dbentryid]}}})
        else:
            if dbname in DBref[chainid]["IDs"].keys():
                DBref[chainid]["IDs"][dbname].append(dbentryid)
            else:
                DBref[chainid]["IDs"].update({dbname: [dbentryid]})
        return None

    def __formatDBref(self, DBref):
        """
        @return: the normalize dict that record DB reference information
        @param
        line: The line to be parsed
        """
        for key in DBref.keys():
            try:
                unp = list(set(DBref[key]['IDs']['UNP']))
                DBref[key]['IDs']['UNP'] = unp
            except:
                DBref[key]['IDs']['UNP'] = []
        return DBref

    def __parseSeqresLine(self, line, seqres, pdb_id):
        """
        @return: None
        @param
        line: The line to be parsed
        seqres: The dict that record the SEQRES primary structure(s) in the PDBFile
        """

        # chain_name = pdb_id +"_" + line[21]
        chain_name = pdb_id + "_" + line[11]  # Zhai yuhao changed 21 to 11.Because 11 is the chain name .
        res = line[19:].replace('\n', '').split()
        seqs = self.__addrseqres(res)
    
        if chain_name in seqres:
            seqres[chain_name] += seqs
        else:
            seqres.update({chain_name: seqs})

            # #print(seqres)
        return None

    def __addrseqres(self, res):
        """
        @return: segment of sequence
        @param
        res: a list of some amino acids in a line
        """
        seqs = ''
        for item in res:
            residue_name = self.__transAA(item)
            seqs = seqs + residue_name
        return seqs

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ####################################################################################################################
    # codes under this is functions that used for finish one task.
    ####################################################################################################################
    def __load_PDB_file(self, fileName):
        """Load a PDB file .
        This function can load a PDB file. But not sure what while happen if the file is too big.
        Args:
            fileName:the full path of PDB to be parsed
        Variables(attributes):
            filelines: the lines PDBfile which is to be parsed
        Used functions:
           None
        Returns:
            filelines: only return a line
        Examples:
            None
        Raises:
            IOError:None
        """
        try:
            with open(fileName) as fh:
                filelines = open(fileName).readlines()  # read file and get all lines
                return filelines
        except EnvironmentError as err:
            print(err)

    def __parse_PDB_ID_Line(self, line):
        """Give the protein an ID.
        This function get Protein when somewhere need an ID .
        Args:
            line: The line to be parsed.[62:66]
        Variables(attributes):
            None
        Used functions:
            None
        Returns:
            chainid :the unique id of the protein .
        Examples:
            "1KMH"
        Raises:
            IOError:None
        """
        chainid = line[62:66].strip()  # 返回第62-66位上的内容
        return chainid
    
    
    
    def main(self,PDBfile):
        """Parse all items and return a dic. """
        pdbbase = PDBParserBase()
	# 1
	# PDB_header = pdbbase.get_header_info(PDBfile)
	# 2
	# PDB_title = pdbbase.get_title_info(PDBfile)
	# 3
	# PDB_compnd = pdbbase.get_compnd_info(PDBfile)
	# 4
	# PDB_source = pdbbase.get_source_info(PDBfile)
	# 5
	# PDB_keywords = pdbbase.get_keywords_info(PDBfile)
	# 6
	# PDB_expdta = pdbbase.get_expdta_info(PDBfile)
	# 7
	# PDB_author = pdbbase.get_author_info(PDBfile)
	# 8
	# PDB_revdat = pdbbase.get_revdat_info(PDBfile)
	# 9
	# PDB_remark2 = pdbbase.get_remark2_info(PDBfile)
	# 10
	# PDB_remark3 = pdbbase.get_remark3_info(PDBfile)
	# 11
	# PDB_seqres = pdbbase.get_seqres_info(PDBfile)
	# 12
	# PDB_cryst1 = pdbbase.get_cryst1_info(PDBfile)
	# 13
	# PDB_origxn = pdbbase.get_origxn_info(PDBfile)
	# 14
	# PDB_scalen = pdbbase.get_scalen_info(PDBfile)
	# 15
	# PDB_master = pdbbase.get_master_info(PDBfile)
	#16
	# PDB_site = pdbbase.get_site_info(PDBfile)
	#17
	# PDB_priStructure = pdbbase.get_sequence_fromATOM(PDBfile)
	#18
	# PDB_terStructure = pdbbase.get_atom_info(PDBfile)
	#19
	# PDB_DBREF = pdbbase.get_reference_betweenPDB(PDBfile)
	#20
	# PDB_seqres = pdbbase.get_sequence_fromSEQ(PDBfile)
    
        pdb_mandatory = {}
        pdb_mandatory["header"] = pdbbase.get_header_info(PDBfile)
        pdb_mandatory["title"] = pdbbase.get_title_info(PDBfile)
        pdb_mandatory["compnd"] = pdbbase.get_compnd_info(PDBfile)
        pdb_mandatory["source"] = pdbbase.get_source_info(PDBfile)
        pdb_mandatory["keywords"] = pdbbase.get_keywords_info(PDBfile)
        pdb_mandatory["expdta"] = pdbbase.get_expdta_info(PDBfile)
        pdb_mandatory["author"] = pdbbase.get_author_info(PDBfile)
        pdb_mandatory["revdat"] = pdbbase.get_revdat_info(PDBfile)
        pdb_mandatory["remark2"] = pdbbase.get_remark2_info(PDBfile)
        pdb_mandatory["remark3"] = pdbbase.get_remark3_info(PDBfile)
        #becaus of remark3's difficulty ,I give up parse it 
        pdb_mandatory["seqres"] = pdbbase.get_seqres_info(PDBfile)
        pdb_mandatory["cryst1"] = pdbbase.get_cryst1_info(PDBfile)
        pdb_mandatory["origxn"] = pdbbase.get_origxn_info(PDBfile)
        pdb_mandatory["scalen"] = pdbbase.get_scalen_info(PDBfile)
        pdb_mandatory["master"] = pdbbase.get_master_info(PDBfile)
        #these are from gjt
        pdb_mandatory["site"] = pdbbase.get_site_info(PDBfile)
        pdb_mandatory["priStructure"] = pdbbase.get_sequence_fromATOM(PDBfile)
        pdb_mandatory["terStructure"] = pdbbase.get_atom_info(PDBfile)
        pdb_mandatory["DBREF"] = pdbbase.get_reference_betweenPDB(PDBfile)
        pdb_mandatory["seqres2"] = pdbbase.get_sequence_fromSEQ(PDBfile)
        return pdb_mandatory


if __name__ == "__main__":
    #print('please input the path of the PDBfile:such as E:\\pdb\\pdb3rum.ent,E:\\pdb\\1a0s.pdb')
    
    main()
    
    #print(pdb_mandatory)
    #print("##########")
    with open("total_pdb.txt", "w") as f:
	    f.write(str(pdb_mandatory))
